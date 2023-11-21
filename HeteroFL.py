#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os
import copy
import numpy as np
import torch.nn as nn
import pickle
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from Text import DatasetLM
from FedLPS import make_model_rate, evaluate
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, train_val_test_image, repackage_hidden
from utils.main_flops_counter import count_training_flops
from Models import all_models
from data.reddit.user_data import data_process
import warnings

warnings.filterwarnings('ignore')

args = args_parser()


def split_model(w_server, local_net):
    idx = OrderedDict()
    local_params = copy.deepcopy(local_net.state_dict())
    for k, v in w_server.items():
        parameter_type = k.split('.')[-1]
        if 'weight' in parameter_type or 'bias' in parameter_type:
            if v.dim() > 1:
                input_size = v.size(1)
                output_size = v.size(0)
                local_input_size = local_params[k].size(1)
                local_output_size = local_params[k].size(0)
                model_input_idx = torch.arange(input_size, device=v.device)
                model_output_idx = torch.arange(output_size, device=v.device)
                if local_input_size != input_size:
                    model_input_idx = model_input_idx[:local_input_size]
                if local_output_size != output_size:
                    model_output_idx = model_output_idx[:local_output_size]
                idx[k] = (model_output_idx, model_input_idx)
            else:
                output_size = v.size(0)
                local_output_size = local_params[k].size(0)
                model_output_idx = torch.arange(output_size, device=v.device)
                if local_output_size != output_size:
                    model_output_idx = model_output_idx[:local_output_size]
                idx[k] = model_output_idx

    for k, v in w_server.items():
        parameter_type = k.split('.')[-1]
        if 'weight' in parameter_type or 'bias' in parameter_type:
            if v.dim() > 1:
                local_params[k] = copy.deepcopy(v[torch.meshgrid(idx[k])])
            else:
                local_params[k] = copy.deepcopy(v[idx[k]])
        else:
            local_params[k] = copy.deepcopy(v)
    return local_params, idx


class Client:

    def __init__(self, args, device, id, train_data, val_data, test_data, local_net, mask_rate=1):
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
        self.roll_rate = mask_rate
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = args.lr

        self.local_epochs = args.local_ep
        self.curr_round = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.local_net = local_net.to(device)

        if self.args.dataset == 'reddit' or self.args.dataset == 'cifar10':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss_func = nn.NLLLoss().to(self.device)
        self.reset_optimizer()

        self.traindata_loader = DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True)
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

    def update_weights(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''
        self.local_net = all_models[self.args.dataset](config, device, self.roll_rate)
        local_params, idx = split_model(w_server, self.local_net)
        self.local_net.load_state_dict(copy.deepcopy(local_params))
        epoch_losses, epoch_acc = [], []

        self.curr_round = round
        if lr:
            self.learning_rate = lr
        self.local_net = self.local_net.to(self.device)
        dl_cost = 0
        for key in local_params:
            dl_cost += torch.numel(local_params[key]) * 32
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

                self.optimizer.step()
                list_loss.append(loss.item())
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                corrent += torch.sum(torch.eq(pred_labels, y)).item()

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        cur_params = copy.deepcopy(self.local_net.state_dict())
        self.local_net.load_state_dict(copy.deepcopy(cur_params))
        train_flops = len(self.traindata_loader) * count_training_flops(copy.deepcopy(self.local_net),
                                                                        self.args) * self.args.local_ep

        ul_cost = 0
        for k in cur_params:
            ul_cost += torch.numel(cur_params[k]) * 32

        ret = dict(state=cur_params,
                   params_idx=idx,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret


def combine(w_glob, w_locals, param_idxs, tmp_counts):
    count = {}
    tmp_counts_cpy = copy.deepcopy(tmp_counts)
    updated_parameters = copy.deepcopy(w_glob)
    for k, v in updated_parameters.items():
        parameter_type = k.split('.')[-1]
        count[k] = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
        for m in range(len(w_locals)):
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if v.dim() > 1:
                    tmp_v[torch.meshgrid(param_idxs[m][k])] += w_locals[m][k]
                    count[k][torch.meshgrid(param_idxs[m][k])] += 1
                    tmp_counts_cpy[k][torch.meshgrid(param_idxs[m][k])] += 1
                else:
                    tmp_v[param_idxs[m][k]] += w_locals[m][k]
                    count[k][param_idxs[m][k]] += 1
                    tmp_counts_cpy[k][param_idxs[m][k]] += 1
            else:
                tmp_v += w_locals[m][k]
                count[k] += 1
                tmp_counts_cpy[k] += 1
        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        tmp_counts = tmp_counts_cpy
    return updated_parameters, tmp_counts

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
    if args.dataset == 'reddit':
        # dataload
        data_dir = './data/reddit/train/'
        with open('./data/reddit/reddit_vocab.pck', 'rb') as f:
            vocab = pickle.load(f)
        nvocab = vocab['size']
        config.nvocab = nvocab
        train_data, val_data, test_data = data_process(data_dir, nvocab, args.nusers)
        for i in range(args.nusers):
            dataset_train[i] = DatasetLM(train_data[i], vocab['vocab'])
            dataset_val[i] = DatasetLM(val_data[i], vocab['vocab'])
            dataset_test[i] = DatasetLM(test_data[i], vocab['vocab'])
            data_num[i] = len(train_data[i])
        all_val, all_test = [], []
        for value in val_data.values():
            for sent in value:
                all_val.append(sent)
        for value in test_data.values():
            for sent in value:
                all_test.append(sent)
    elif args.dataset == 'mnist':
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

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_HeteroFL_{}.pt'.format(args.dataset, args.seed)
    config.mask_weight_indicator = []
    config.personal_layers = []

    # initialized global model
    net_glob = all_models[args.dataset](config, device)
    net_glob = net_glob.to(device)
    w_glob = net_glob.state_dict()
    tmp_counts = {}
    for k, v in w_glob.items():
        tmp_counts[k] = torch.ones_like(v)
    # initialize clients
    clients = {}
    client_ids = []
    config.mask_rate_list = [0.0625, 0.125, 0.25, 0.5, 1]
    config.proportion = [1, 1, 1, 1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    clients_mask_rate = np.array(config.mask_rate_list)[rate_idx]
    for client_id in range(args.nusers):
        cl = Client(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                    dataset_test[client_id],
                    local_net=all_models[args.dataset](config, device), mask_rate=clients_mask_rate[client_id])
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    # we need to accumulate compute/DL/UL costs regardless of round number, resetting only
    # when we actually report these numbers
    compute_flops = np.zeros(len(clients))  # the total floating operations for each client
    download_cost = np.zeros(len(clients))
    upload_cost = np.zeros(len(clients))

    for round in range(args.rounds):
        clients_mask_rate = make_model_rate(config, clients_mask_rate)
        upload_cost_round = []
        download_cost_round = []
        compute_flops_round = []

        net_glob.train()
        w_locals, param_idxs, avg_weight, loss_locals, acc_locals = [], [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            client.roll_rate = clients_mask_rate[i]
            train_result = client.update_weights(w_server=copy.deepcopy(w_glob), round=round)
            w_locals.append(train_result['state'])
            param_idxs.append(train_result['params_idx'])
            avg_weight.append(data_num[idx])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost[i] += train_result['dl_cost']
            download_cost_round.append(train_result['dl_cost'])
            upload_cost[i] += train_result['ul_cost']
            upload_cost_round.append(train_result['ul_cost'])
            compute_flops[i] += train_result['train_flops']
            compute_flops_round.append(train_result['train_flops'])

        w_glob, tmp_counts = combine(w_glob, w_locals, param_idxs, tmp_counts)
        net_glob.load_state_dict(w_glob)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        per_test_loss, per_test_acc = [], []
        for _, c in clients.items():
            per_train_res = evaluate(config, c.traindata_loader, net_glob, device)
            train_loss.append(per_train_res[0])
            train_acc.append(per_train_res[1])
            per_val_res = evaluate(config, c.valdata_loader, net_glob, device)
            val_loss.append(per_val_res[0])
            val_acc.append(per_val_res[1])
            per_test_res = evaluate(config, c.testdata_loader, net_glob, device)
            per_test_loss.append(per_test_res[0])
            per_test_acc.append(per_test_res[1])
        print('\nTrain loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                    sum(train_acc) / len(train_acc)))
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))

        print("Global Round {}, Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(round, sum(val_loss) / len(val_loss),
                                            sum(val_acc) / len(val_acc)))
        print("Global Round {}, test loss: {:.5f}, "
              "test accuracy: {:.5f}".format(round, sum(per_test_loss) / len(per_test_loss),
                                             sum(per_test_acc) / len(per_test_acc)))