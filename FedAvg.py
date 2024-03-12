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
    get_dataset_tiny_extr_noniid, train_val_test_image, repackage_hidden
from utils.main_flops_counter import count_training_flops
from dataset import DatasetSplit
from Models import all_models
from data.reddit.user_data import data_process
from Client import Client, evaluate
import warnings

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]


def make_capacity(config, client_capacity):
    if config.model_split_mode == 'dynamic':
        rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                     replacement=True).tolist()
        client_capacity = np.array(FLOPS_capacity)[rate_idx]
    elif config.model_split_mode == 'fix':
        client_capacity = np.array(client_capacity)
    else:
        raise ValueError('Not valid model split mode')
    return client_capacity


if __name__ == "__main__":
    args.fedprox = False
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

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_FedAvg_{}.pt'.format(args.dataset, args.seed)
    config.mask_weight_indicator = []
    config.personal_layers = []

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
        w_locals, avg_weight, loss_locals, acc_locals = [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            train_result = client.update_weights(w_server=copy.deepcopy(w_glob), round=round)
            w_locals.append(train_result['state'])
            avg_weight.append(data_num[idx])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

        w_glob = average_weights(w_locals, avg_weight, config)
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
        print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, sum(train_loss) / len(train_loss),
                                                                    sum(train_acc) / len(train_acc)), flush=True)
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)))

        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(sum(val_loss) / len(val_loss),
                                            sum(val_acc) / len(val_acc)))
        print("test loss: {:.5f}, "
              "test accuracy: {:.5f}".format(sum(per_test_loss) / len(per_test_loss),
                                             sum(per_test_acc) / len(per_test_acc)), flush=True)