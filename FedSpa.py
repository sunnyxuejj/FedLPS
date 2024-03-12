#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import copy
import os

import pickle

from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image
from utils.main_flops_counter import count_training_flops
from util import repackage_hidden
from Models import all_models, needs_mask
from Client import Client, evaluate
from FedAvg import make_capacity
from data.reddit.user_data import data_process
import warnings

warnings.filterwarnings('ignore')

args = args_parser()


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


def prune_by_layer(test_model, args, masks, sparsity=0.4, sparsity_distribution='erk'):
    step = 0
    weights_by_layer = test_model._weights_by_layer(sparsity=sparsity,
                                                    sparsity_distribution=sparsity_distribution)
    mask_half = []
    for name, param in test_model.named_parameters():
        # We need to figure out how many to prune
        n_total = 0
        if needs_mask(name, args.mask_weight_indicator):
            n_total += param.numel()
            n_prune = int(n_total - weights_by_layer[name])
            if n_prune >= n_total or n_prune < 0:
                continue
                # Determine smallest indices
            if not torch.where(param == 1.0, True, False).all():
                _, prune_indices = torch.topk(torch.abs(param.data.flatten()),
                                              n_prune, largest=False)
            else:
                prune_indices = torch.tensor(list(range(param.shape[0])[-n_prune:]))

            # Write mask
            param.data.view(param.data.numel())[prune_indices] = 0
            _mask = torch.from_numpy(masks[step])
            _mask.view(_mask.numel())[prune_indices] = 0
            mask_half.append(_mask.cpu().numpy())
            step += 1
    return mask_half


class Client_FedSpa(Client):
    def __init__(self, *args, **kwargs):
        super(Client_FedSpa, self).__init__(*args, **kwargs)
        self.mask = make_mask(copy.deepcopy(self.local_net))

    def prune_by_dst(self, round, sparsity=0.4, sparsity_distribution='erk'):
        test_model = copy.deepcopy(self.local_net)
        optimizer = torch.optim.SGD(test_model.parameters(),
                                    lr=self.args.lr * (self.args.lr_decay ** round),
                                    momentum=self.args.momentum, weight_decay=self.args.wdecay)
        test_model.train()
        for batch_ind, local_data in enumerate(self.traindata_loader):
            optimizer.zero_grad()

            if self.args.dataset == 'reddit':
                x = torch.stack(local_data[:-1]).to(self.device)
                y = torch.stack(local_data[1:]).view(-1).to(self.device)
                hidden = test_model.init_hidden(self.args.local_bs)
                hidden = repackage_hidden(hidden)
                if hidden[0][0].size(1) != x.size(1):
                    hidden = test_model.init_hidden(x.size(1))
                out, hidden = test_model(x, hidden)
            else:
                x, y = local_data[0].to(self.device), local_data[1].to(self.device)
                out = test_model(x)
            self.loss_func(out, y).backward()

            mask_half = prune_by_layer(test_model, args, copy.deepcopy(self.mask), sparsity, sparsity_distribution)

            weights_by_layer = test_model._weights_by_layer(sparsity=0.5,
                                                            sparsity_distribution=sparsity_distribution)
            step = 0
            for name, param in test_model.named_parameters():
                if not needs_mask(name, test_model.mask_weight_indicator):
                    continue
                # We need to figure out how many to grow
                _mask = torch.from_numpy(mask_half[step])
                n_nonzero = (_mask != 0).float().sum()
                n_grow = int(weights_by_layer[name] - n_nonzero)
                if n_grow < 0:
                    continue
                # print('grow from', n_nonzero, 'to', weights_by_layer[name])

                _, grow_indices = torch.topk(torch.abs(param.grad.flatten()),
                                             n_grow, largest=True)

                # Write and apply mask
                param.data.view(param.data.numel())[grow_indices] = 0
                _mask = torch.from_numpy(mask_half[step])
                _mask.view(_mask.numel())[grow_indices] = 1
                self.mask[step] = _mask.cpu().numpy()
                step += 1
            break

        step = 0
        for name, param in self.local_net.named_parameters():

            # We do not prune bias term
            if name in self.local_net.mask_weight_indicator:
                weight_dev = param.device
                param.data = param.data * torch.from_numpy(self.mask[step]).to(weight_dev)
                step += 1

    def train(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''
        EPS = 1e-9
        epoch_losses, epoch_acc = [], []
        if lr:
            self.learning_rate = lr
        a = self.local_net.state_dict()
        self.local_net.load_state_dict(copy.deepcopy(w_server))
        self.local_net = self.local_net.to(self.device)
        dl_cost = self.local_net.stat_param_sizes()
        val_res = evaluate(self.args, self.valdata_loader, copy.deepcopy(self.local_net), self.device)
        acc_beforeTrain = val_res[1]
        if acc_beforeTrain > self.args.prune_start_acc:
            self.mask = prune_by_layer(self.local_net, args, self.mask, self.args.sparsity)
            self.prune_by_dst(round)

        self.local_net.train()

        self.reset_optimizer(round=round)

        for iter in range(self.args.local_ep):
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

        # if acc_beforeTrain > self.args.prune_start_acc:
        # self.mask = prune_by_layer(self.local_net, args, self.mask, self.args.sparsity)
        # self.prune_by_dst(round)

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
    model_saved = './log/{}/model_FedSpa_{}.pt'.format(args.dataset, args.seed)
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
    FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]
    config.proportion = [1, 1, 1, 1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(args.nusers):
        cl = Client_FedSpa(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
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
        print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, sum(train_loss) / len(train_loss),
                                                                              sum(train_acc) / len(train_acc)),
              flush=True)
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)))

        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(sum(per_val_loss) / len(per_val_loss),
                                            sum(per_val_acc) / len(per_val_acc)), flush=True)
        print("test loss: {:.5f}, test accuracy: {:.5f}".format(sum(glob_test_loss) / len(glob_test_loss),
                                                                sum(glob_test_acc) / len(glob_test_acc)), flush=True)
