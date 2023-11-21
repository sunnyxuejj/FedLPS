#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os
import math
import pickle
import pandas as pd

from torch.utils.data import DataLoader
from Text import DatasetLM
from Hetero_Client import Hetero_Client, evaluate
from agg.avg import *
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, train_val_test_image
from data.reddit.user_data import data_process
from Models import all_models
import warnings
warnings.filterwarnings('ignore')

args = args_parser()


def U_function(x):
    return 10 - 20 / (1 + math.exp(0.35 * x))


def make_model_rate(config, clients_model_rate):
    if config.model_split_mode == 'dynamic':
        rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                     replacement=True).tolist()
        clients_model_rate = np.array(config.mask_rate_list)[rate_idx]
    elif config.model_split_mode == 'fix':
        clients_model_rate = np.array(clients_model_rate)
    else:
        raise ValueError('Not valid model split mode')
    return clients_model_rate


if __name__ == "__main__":
    config = args
    args.mask = True
    args.learnable = True

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
    model_saved = './log/{}/model_FedLPS_{}.pt'.format(args.dataset, args.seed)

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
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    clients_mask_rate = np.array(config.mask_rate_list)[rate_idx]
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

    try:
        for round in range(args.rounds):
            clients_mask_rate = make_model_rate(config, clients_mask_rate)
            upload_cost_round = []
            download_cost_round = []
            compute_flops_round = []

            net_glob.train()
            w_locals, loss_locals, acc_locals = [], [], []
            m = max(int(args.frac * len(clients)), 1)
            idxs_users = np.random.choice(len(clients), m, replace=False) # 随机采样client
            total_num = 0
            for idx in idxs_users:
                client = clients[idx]
                i = client_ids.index(idx)
                client.mask_rate = clients_mask_rate[i]
                mab_mask_rate, action_index = client.P_UCBV()
                client.mask_rate = min(client.mask_rate, mab_mask_rate)
                train_result = client.update_weights_leanable(w_server=copy.deepcopy(w_glob), round=round)
                w_locals.append(train_result['state'])
                loss_locals.append(train_result['loss'])
                acc_locals.append(train_result['acc'])

                download_cost[i] += train_result['dl_cost']
                download_cost_round.append(train_result['dl_cost'])
                upload_cost[i] += train_result['ul_cost']
                upload_cost_round.append(train_result['ul_cost'])
                compute_flops[i] += train_result['train_flops']
                compute_flops_round.append(train_result['train_flops'])

                # P-UCBV
                # reddit
                reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                         / (3 * train_result['train_flops'] / 1e6 / 67200 / clients_mask_rate[i]
                            + train_result['ul_cost'] / 8 / 1024 / 1024 / 14.9)
                # cifar10
                # reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                #          / (3 * train_result['train_flops'] / 1e6 / 672000 / clients_mask_rate[i]
                #             + train_result['ul_cost'] / 8 / 1024 / 1024 / 14.9)
                # mnist
                # reward = (U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))) \
                #          / (3 * train_result['train_flops'] / 1e6 / 2788.128 / clients_mask_rate[i]
                #             + train_result['ul_cost'] / 8 / 1024 / 1024)
                # reward = U_function(client.acc_record[-1]) - U_function(max(client.acc_record[:-1]))
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

            w_glob = avg(w_locals, w_glob, args, device)
            if config.dataset == 'reddit' and config.tie_weights:
                w_glob[last_layer] = copy.deepcopy(w_glob[first_layer])
            net_glob.load_state_dict(w_glob)
            net_glob = net_glob.to(device)

            train_loss, train_acc, val_loss, val_acc = [], [], [], []
            for _, c in clients.items():
                local_model = copy.deepcopy(net_glob)
                if c.final_parameters is not None:
                    if c.rec_mask_rate is not None and c.rec_mask_rate > config.mask_rate_list[2]:
                        local_params = copy.deepcopy(c.final_parameters)
                    else:
                        local_params = copy.deepcopy(local_model.state_dict())
                        for key in config.personal_layers:
                            local_params[key] = copy.deepcopy(c.final_parameters[key])
                    local_model.load_state_dict(local_params)
                train_res = evaluate(config, c.traindata_loader, local_model, device)
                per_val_res = evaluate(config, c.valdata_loader, local_model, device)
                train_loss.append(train_res[0])
                train_acc.append(train_res[1])
                val_loss.append(per_val_res[0])
                val_acc.append(per_val_res[1])
            print('\nTrain loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                        sum(train_acc) / len(train_acc)))
            print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
                  'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
            print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
            print("Personalized Round {}, Validation loss: {:.5f}, "
                  "val accuracy: {:.5f}".format(round, sum(val_loss) / len(val_loss),
                                                sum(val_acc) / len(val_acc)))

            if not best_val_acc or sum(val_acc) / len(val_acc) > best_val_acc:
                with open(model_saved, 'wb') as f:
                    print('save model')
                    torch.save(net_glob, f)
                best_val_acc = sum(val_acc) / len(val_acc)

            per_test_loss, per_test_acc = [], []
            for _, c in clients.items():
                local_model = copy.deepcopy(net_glob)
                if c.final_parameters is not None:
                    if c.rec_mask_rate is not None and c.rec_mask_rate > config.mask_rate_list[2]:
                        local_params = copy.deepcopy(c.final_parameters)
                    else:
                        local_params = copy.deepcopy(local_model.state_dict())
                        for key in config.personal_layers:
                            local_params[key] = copy.deepcopy(c.final_parameters[key])
                    local_model.load_state_dict(local_params)
                per_test_res = evaluate(config, c.testdata_loader, local_model, device)
                per_test_loss.append(per_test_res[0])
                per_test_acc.append(per_test_res[1])
            print("Personalized Round {}, test loss: {:.5f}, "
                  "test accuracy: {:.5f}".format(round, sum(per_test_loss) / len(per_test_loss),
                                                 sum(per_test_acc) / len(per_test_acc)))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Existing from training early')

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    per_test_loss, per_test_acc = [], []
    for _, c in clients.items():
        local_model = copy.deepcopy(model_best)
        if c.final_parameters is not None:
            if c.rec_mask_rate is not None and c.rec_mask_rate > config.mask_rate_list[2]:
                local_params = copy.deepcopy(c.final_parameters)
            else:
                local_params = copy.deepcopy(local_model.state_dict())
                for key in config.personal_layers:
                    local_params[key] = copy.deepcopy(c.final_parameters[key])
            local_model.load_state_dict(local_params)
        per_test_res = evaluate(config, c.testdata_loader, local_model, device)
        per_test_loss.append(per_test_res[0])
        per_test_acc.append(per_test_res[1])
    print("Personalized test loss: {:.5f}, "
          "test accuracy: {:.5f}".format(sum(per_test_loss) / len(per_test_loss),
                                         sum(per_test_acc) / len(per_test_acc)))