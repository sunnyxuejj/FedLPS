#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os
import math
import pickle
import pandas as pd

from torch.utils.data import DataLoader
from torchtext import data
from Hetero_Client import Hetero_Client
from agg.avg import *
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image, repackage_hidden
from dataset import DatasetSplit
from data.reddit.user_data import data_process
from Models import all_models
from Client import evaluate
import warnings

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]
# FLOPS_capacity = [186112.0, 372224.0, 744448.0]
# FLOPS_capacity = [372224.0, 744448.0]


def U_function(x):
    return 10 - 20 / (1 + math.exp(0.35 * x))


def make_model_rate(config, clients_model_rate, client_capacity):
    if config.model_split_mode == 'dynamic':
        rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                     replacement=True).tolist()
        clients_model_rate = np.array(config.mask_rate_list)[rate_idx]
        client_capacity = np.array(FLOPS_capacity)[rate_idx]
    elif config.model_split_mode == 'fix':
        clients_model_rate = np.array(clients_model_rate)
        client_capacity = np.array(client_capacity)
    else:
        raise ValueError('Not valid model split mode')
    return clients_model_rate, client_capacity


if __name__ == "__main__":
    config = args
    config.mask = True
    config.learnable = True
    config.personalized = True
    config.online_decision = True
    args.strategy = 'FedLPS'

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
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar10':
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
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar100':
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
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'tinyimagenet':
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
            data_num[i] = len(dataset_train[i])

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_{}_{}.pt'.format(args.dataset, args.strategy, args.seed)

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
    config.personal_layers.append(layers[-2])
    config.personal_layers.append(layers[-1])

    model_indicator.to(device)
    if args.mask:
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
    config.mask_rate_list = [0.0625, 0.125, 0.25, 0.5, 1]
    config.proportion = [1, 1, 1, 1, 1]
    # config.mask_rate_list = [0.25, 0.5, 1]
    # config.proportion = [1, 1, 1]
    # config.mask_rate_list = [0.5, 1]
    # config.proportion = [1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    clients_mask_rate = np.array(config.mask_rate_list)[rate_idx]
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(args.nusers):
        cl = Hetero_Client(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
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
        clients_mask_rate, client_capacity = make_model_rate(config, clients_mask_rate, client_capacity)
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []

        net_glob.train()
        w_locals, loss_locals, acc_locals = [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            client.mask_rate = clients_mask_rate[i]
            if args.online_decision:
                mab_mask_rate, action_index = client.P_UCBV()
                client.mask_rate = min(client.mask_rate, mab_mask_rate)
            train_result = client.update_weights_leanable(w_server=copy.deepcopy(w_glob), round=round)
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

            if args.dataset == 'reddit':
                reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                         / (3 * train_result['train_flops'] / 1e6 / 67200 / clients_mask_rate[i]
                            + train_result['ul_cost'] / 8 / 1024 / 1024 / 14.9)
            # cifar10 / cifar100 / tinyimagenet
            elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
                reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                         / (3 * train_result['train_flops'] / 1e6 / 672000 / clients_mask_rate[i]
                            + train_result['ul_cost'] / 8 / 1024 / 1024 / 14.9)
            # mnist
            elif args.dataset == 'mnist':
                reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                         / (3 * train_result['train_flops'] / 1e6 / 2788.128 / clients_mask_rate[i]
                            + train_result['ul_cost'] / 8 / 1024 / 1024)

            action_index = 0
            for i, n in enumerate(client.mab_arms):
                if n[0] > client.mask_rate:
                    action_index = i - 1
                    break
            client.arms_reward[client.mab_arms[action_index][0]].append(reward.item())
            if client.acc_record[-1] - client.acc_record[-2] < -5:
                client.mab_arms = np.delete(client.mab_arms, action_index, axis=0)
                client.pull_times = np.delete(client.pull_times, action_index)
                del client.arms_reward[client.mab_arms[action_index][0]]

        if args.agg == 'avg':
            w_glob = avg(w_locals, w_glob, args, device)
        if config.dataset == 'reddit':
            w_glob[last_layer] = copy.deepcopy(w_glob[first_layer])
        net_glob.load_state_dict(w_glob)
        net_glob = net_glob.to(device)

        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        per_test_loss, per_test_acc = [], []
        for _, c in clients.items():
            local_model = copy.deepcopy(net_glob)
            if c.final_parameters is not None:
                if c.rec_mask_rate is not None and c.rec_mask_rate > config.mask_rate_list[1]:
                    local_params = copy.deepcopy(c.final_parameters)
                else:
                    local_params = copy.deepcopy(local_model.state_dict())
                    for key in config.personal_layers:
                        local_params[key] = copy.deepcopy(c.final_parameters[key])
                local_model.load_state_dict(local_params)
                # local_model.load_state_dict(c.final_parameters)
            train_res = evaluate(config, c.traindata_loader, local_model, device)
            per_val_res = evaluate(config, c.valdata_loader, local_model, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            val_loss.append(per_val_res[0])
            val_acc.append(per_val_res[1])
            per_test_res = evaluate(config, c.testdata_loader, local_model, device)
            per_test_loss.append(per_test_res[0])
            per_test_acc.append(per_test_res[1])
        print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, sum(train_loss) / len(train_loss),
                                                                              sum(train_acc) / len(train_acc)),
              flush=True)
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)), flush=True)
        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(sum(val_loss) / len(val_loss),
                                            sum(val_acc) / len(val_acc)))
        print("test loss: {:.5f}, "
              "test accuracy: {:.5f}".format(sum(per_test_loss) / len(per_test_loss),
                                             sum(per_test_acc) / len(per_test_acc)), flush=True)
