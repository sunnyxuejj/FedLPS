#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os

import torch.nn as nn
import pickle

from torch.utils.data import DataLoader
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image
from utils.main_flops_counter import count_training_flops
from util import repackage_hidden
from Models import all_models
from data.reddit.user_data import data_process
import warnings
from Client import Client, evaluate
from FedAvg import make_capacity

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]


def average_weights_with_masks(w, masks, config):
    '''
    Returns the average of the weights computed with masks.
    '''
    step = 0
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if key in config.mask_weight_indicator:
            mask = masks[0][step]
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                mask += masks[i][step]
            w_avg[key] = torch.from_numpy(np.where(mask < 1, 0, w_avg[key].cpu().numpy() / mask)).to(config.device)
            step += 1
        else:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def mix_global_weights(global_weights_last, global_weights_epoch, masks, config):
    step = 0
    global_weights = copy.deepcopy(global_weights_epoch)
    for key in global_weights.keys():
        if key in config.mask_weight_indicator:
            mask = masks[0][step]
            for i in range(1, len(masks)):
                mask += masks[i][step]
            global_weights[key] = torch.from_numpy(
                np.where(mask < 1, global_weights_last[key].cpu(), global_weights_epoch[key].cpu())).to(config.device)
            step += 1
    return global_weights


def make_mask(model):
    mask_layers_name = model.mask_weight_indicator
    mask = [None] * len(mask_layers_name)
    step = 0
    for name, param in model.named_parameters():
        if name in mask_layers_name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    return mask


def mask_model(model, mask):
    step = 0
    for name, param in model.named_parameters():
        if name in model.mask_weight_indicator:
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * param.data.cpu().numpy()).to(weight_dev)
            step = step + 1


class Client:

    def __init__(self, args, device, id, train_data, val_data, test_data, local_net):
        '''Construct a new client.

        Parameters:
        args:
            related parameters settings
        device: 'cpu' or 'cuda'
            running device label
        id : object
            a unique identifier for this client.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        val_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us validation samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        initial_global_params : dict
            initial global model parameters
        lr: float
            current learning rate

        Returns: a new client.
        '''

        self.args = args
        self.device = device
        self.id = id
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.criterion = nn.CrossEntropyLoss()
        self.select_time = 0

        self.learning_rate = args.lr

        self.local_epochs = args.local_ep
        self.curr_round = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.local_net = local_net.to(device)
        self.mask = make_mask(copy.deepcopy(self.local_net))
        self.pruning_rate = 1

        if self.args.dataset == 'reddit' or self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100' or self.args.dataset == 'tinyimagenet':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss_func = nn.NLLLoss().to(self.device)
        self.l1_penalty = nn.L1Loss().to(self.device)
        self.l2_penalty = nn.MSELoss().to(self.device)
        self.reset_optimizer()

        self.traindata_loader = DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True)
        if args.dataset == 'reddit':
            self.valdata_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            self.testdata_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        else:
            self.valdata_loader = DataLoader(val_data, batch_size=self.args.local_bs, shuffle=True)
            self.testdata_loader = DataLoader(test_data, batch_size=self.args.local_bs, shuffle=True)

    def reset_optimizer(self, round=0):
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.local_net.parameters()),
                                             lr=self.args.lr * (self.args.lr_decay ** round),
                                             momentum=self.args.momentum, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_net.parameters()),
                                              lr=self.learning_rate, weight_decay=self.args.wdecay)

    def prune_by_percentile(self):
        # Calculate percentile value
        step = 0
        for name, param in self.local_net.named_parameters():

            # We do not prune bias term
            if name in self.local_net.mask_weight_indicator:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), self.args.prune_percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, self.mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[step] = new_mask
                step += 1

    def train(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''
        self.select_time += 1
        EPS = 1e-6
        epoch_losses, epoch_acc = [], []
        self.curr_round = round
        if lr:
            self.learning_rate = lr
        self.local_net.load_state_dict(copy.deepcopy(w_server))
        self.local_net = self.local_net.to(self.device)
        dl_cost = self.local_net.stat_param_sizes()
        val_res = evaluate(config, self.valdata_loader, copy.deepcopy(self.local_net), self.device)
        acc_beforeTrain = val_res[1]
        if (acc_beforeTrain > self.args.prune_start_acc and self.pruning_rate > self.args.prune_end_rate):
            self.prune_by_percentile()
            self.pruning_rate = self.local_net.stat_param_sizes() / self.local_net.param_size
        self.local_net.train()

        self.reset_optimizer(round=round)

        for iter in range(self.local_epochs):
            list_loss = []
            total, corrent = 0, 0
            for batch_ind, local_data in enumerate(self.traindata_loader):
                self.optimizer.zero_grad()

                if self.args.dataset == 'reddit':
                    x = torch.stack(local_data[:-1]).to(self.device)
                    y = torch.stack(local_data[1:]).view(-1).to(self.device)
                    total += y.size(0)
                    hidden = self.local_net.init_hidden(self.args.local_bs)
                    hidden = repackage_hidden(hidden)
                    if hidden[0][0].size(1) != x.size(1):
                        hidden = self.local_net.init_hidden(x.size(1))
                    out, hidden = self.local_net(x, hidden)
                else:
                    x, y = local_data[0].to(self.device), local_data[1].to(self.device)
                    total += len(y)
                    out = self.local_net(x)

                loss = self.loss_func(out, y)
                loss.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.args.clip)
                for name, p in self.local_net.named_parameters():
                    if name in self.local_net.mask_weight_indicator:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        grad_tensor = np.where(abs(tensor) < EPS, 0, grad_tensor)
                        p.grad.data = torch.from_numpy(grad_tensor).to(device)

                self.optimizer.step()
                list_loss.append(loss.item())
                if self.args.dataset == 'reddit':
                    top_3, top3_index = torch.topk(out, 3, dim=1)
                    for i in range(top3_index.size(0)):
                        if y[i] in top3_index[i]:
                            corrent += 1
                else:
                    _, pred_labels = torch.max(out, 1)
                    pred_labels = pred_labels.view(-1)
                    corrent += torch.sum(torch.eq(pred_labels, y)).item()

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        ul_cost = self.local_net.stat_param_sizes()
        cur_params = copy.deepcopy(self.local_net.state_dict())
        train_flops = len(self.traindata_loader) * count_training_flops(copy.deepcopy(self.local_net),
                                                                        self.args, full=False) * self.args.local_ep
        grad = {}
        step = 0
        for k in cur_params:
            if k in self.local_net.mask_weight_indicator:
                cur_params[k] = cur_params[k] * torch.from_numpy(self.mask[step]).to(self.device)
                w_server[k] = w_server[k] * torch.from_numpy(self.mask[step]).to(self.device)
                step += 1
            grad[k] = cur_params[k] - w_server[k]

        ret = dict(state=grad,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret


if __name__ == "__main__":
    config = args

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu != -1:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    config.device = device

    dataset_train, dataset_val, dataset_test, data_num = {}, {}, {}, {}
    if args.dataset == 'mnist':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_mnist_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar10':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar10_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar100':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar100_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'reddit':
        # dataload
        data_dir = 'data/reddit/train/'
        with open('data/reddit/reddit_vocab.pck', 'rb') as f:
            vocab = pickle.load(f)
        nvocab = vocab['size']
        config.nvocab = nvocab
        train_data, val_data, test_data = data_process(data_dir, nvocab, args.nusers)
        for i in range(args.nusers):
            dataset_train[i] = DatasetLM(train_data[i], vocab['vocab'])
            dataset_val[i] = DatasetLM(val_data[i], vocab['vocab'])
            dataset_test[i] = DatasetLM(test_data[i], vocab['vocab'])
            data_num[i] = len(train_data[i])
    elif args.dataset == 'tinyimagenet':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_tiny_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])

    best_val_loss = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_LotteryFL_{}.pt'.format(args.dataset, args.seed)
    config.personal_layers = []

    # 确定哪些层可以被mask
    config.mask_weight_indicator = []
    config.personal_layers = []
    model_indicator = all_models[args.dataset](config, device)
    model_weight = copy.deepcopy(model_indicator.state_dict())
    layers = list(model_weight.keys())
    layers_name = []
    for key in layers:
        if 'weight' in key:
            layers_name.append(key)
    first_layer = layers_name[0]
    last_layer = layers_name[-1]
    model_indicator.to(device)
    model_indicator.label_mask_weight()
    mask_weight_indicator = model_indicator.mask_weight_indicator
    if last_layer in mask_weight_indicator:
        mask_weight_indicator = mask_weight_indicator[:-1]
    config.mask_weight_indicator = copy.deepcopy(mask_weight_indicator)

    # initialized global model
    net_glob = all_models[args.dataset](config, device)
    net_glob = net_glob.to(device)
    w_glob = net_glob.state_dict()
    # initialize clients
    clients = {}
    client_ids = []
    config.proportion = [1, 1, 1, 1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(args.nusers):
        cl = Client(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                    dataset_test[client_id],
                    local_net=all_models[args.dataset](config, device))
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    for round in range(args.rounds):
        client_capacity = make_capacity(config, client_capacity)
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []

        net_glob.train()
        w_locals, mask_locals, loss_locals, acc_locals = [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            train_model = copy.deepcopy(net_glob)
            mask_model(train_model, client.mask)
            train_result = client.train(w_server=copy.deepcopy(train_model.state_dict()), round=round)
            w_locals.append(train_result['state'])
            mask_locals.append(client.mask)
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

        # global_weights_epoch = average_weights_with_masks(w_locals, mask_locals, config)
        # w_glob = mix_global_weights(w_glob, global_weights_epoch, mask_locals, config)
        w_glob = avg(w_locals, w_glob, config, device)
        net_glob.load_state_dict(w_glob)
        net_glob = net_glob.to(device)

        train_loss, train_acc = [], []
        per_val_loss, per_val_acc = [], []
        glob_test_loss, glob_test_acc = [], []
        for _, c in clients.items():
            local_mask_model = copy.deepcopy(net_glob)
            mask_model(local_mask_model, c.mask)
            train_res = evaluate(config, c.traindata_loader, copy.deepcopy(local_mask_model), device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            per_val_res = evaluate(config, c.valdata_loader, copy.deepcopy(local_mask_model), device)
            per_val_loss.append(per_val_res[0])
            per_val_acc.append(per_val_res[1])
            glob_test_res = evaluate(config, c.testdata_loader, copy.deepcopy(local_mask_model), device)
            glob_test_loss.append(glob_test_res[0])
            glob_test_acc.append(glob_test_res[1])
        print(
            '\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, sum(train_loss) / len(per_val_loss),
                                                                            sum(train_acc) / len(train_acc)))
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6), flush=True)

        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)))
        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(sum(per_val_loss) / len(per_val_loss),
                                            sum(per_val_acc) / len(per_val_acc)), flush=True)
        print("test loss: {:.5f}, test accuracy: {:.5f}".format(sum(glob_test_loss) / len(glob_test_loss),
                                                                sum(glob_test_acc) / len(glob_test_acc)), flush=True)
