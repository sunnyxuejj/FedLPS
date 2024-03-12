# -*- coding: utf-8 -*-
# @python: 3.7
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from agg.avg import *
from torch.utils.data import DataLoader
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image, repackage_hidden
from utils.main_flops_counter import count_training_flops
from dataset import DatasetSplit
from Models import all_models, needs_mask
from data.reddit.user_data import data_process
from Client import Client, evaluate
from FedAvg import make_capacity
import warnings

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]


def initialize_mask_layer(layer, dtype=torch.bool):
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.LSTM) or isinstance(layer, torch.nn.Linear) or \
            isinstance(layer, torch.nn.Embedding) or isinstance(layer, torch.nn.GRU):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                if hasattr(layer, name + '_mask'):
                    warnings.warn(
                        'Parameter has a pruning mask already. '
                        'Reinitialize to an all-one mask.'
                    )
                layer.register_buffer(name + '_mask', torch.ones_like(param, dtype=dtype))


def initialize_mask(model, dtype=torch.bool):
    layers_to_prune = (layer for _, layer in model.named_children())
    for layer in layers_to_prune:
        if isinstance(layer, torch.nn.Sequential):
            layers = [layer[i] for i in range(len(layer))]
            for layer_ in layers:
                initialize_mask_layer(layer_)
        elif isinstance(layer, torch.nn.ModuleList):
            layers = [layer[i] for i in range(len(layer))]
            for layer_ in layers:
                initialize_mask_layer(layer_)
        else:
            initialize_mask_layer(layer)


def count_layerwise_params_num(model, config):
    layer_times = []
    for name, param in model.named_parameters():
        if needs_mask(name, config.mask_weight_indicator):
            layer_times.append(torch.numel(param.data))
    return layer_times


class Client_PruneFL(Client):
    def __init__(self, *args, **kwargs):
        super(Client_PruneFL, self).__init__(*args, **kwargs)
        initialize_mask(self.local_net)

    def train_size(self):
        return sum(len(x) for x in self.train_data)

    def reset_weights(self, *args, **kwargs):
        return self.local_net.reset_weights(*args, **kwargs)

    def train(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0
        train_flops = 0
        epoch_losses, epoch_acc = [], []

        mask_changed = self.reset_weights(global_state=w_server, use_global_mask=True)  # 判断mask是否改变

        # Try to reset the optimizer state.
        self.reset_optimizer(round)

        if mask_changed:
            dl_cost += self.local_net.mask_size  # need to receive mask
        dl_cost += self.local_net.stat_param_sizes()

        # pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        for iter in range(self.args.local_ep):

            self.local_net.train()

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
                self.reset_weights()  # applies the mask

            train_flops += len(self.traindata_loader) * count_training_flops(copy.deepcopy(self.local_net),
                                                                             self.args)

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

        # we only need to transmit the masked weights and all biases
        ul_cost += self.local_net.stat_param_sizes() + self.local_net.mask_size

        ret = dict(state=copy.deepcopy(self.local_net.state_dict()),
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret


if __name__ == "__main__":
    if args.rate_decay_end is None:
        args.rate_decay_end = args.rounds // 2
    if args.final_sparsity is None:
        args.final_sparsity = args.sparsity
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
        val_list = [i for item in idx_vals for i in item]
        loader_val = DataLoader(DatasetSplit(total_train_data, val_list), batch_size=args.bs, shuffle=True)
        loader_test = DataLoader(total_test_data, batch_size=args.bs, shuffle=True)

    best_val_loss = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_PruneFL_{}.pt'.format(args.dataset, args.seed)

    # 确定哪些层可以被mask
    config.mask_weight_indicator = []
    config.personal_layers = []
    model_indicator = all_models[config.dataset](config, device)
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
    # if first_layer in mask_weight_indicator:
    #     mask_weight_indicator = mask_weight_indicator[1:]
    # if last_layer in mask_weight_indicator:
    #     mask_weight_indicator = mask_weight_indicator[:-1]
    config.mask_weight_indicator = copy.deepcopy(mask_weight_indicator)

    # initialized global model
    net_glob = all_models[args.dataset](config, device)
    net_glob = net_glob.to(device)
    layer_times = count_layerwise_params_num(copy.deepcopy(net_glob), config)
    initialize_mask(net_glob, dtype=torch.bool)
    params = net_glob.state_dict()
    net_glob.layer_prune(sparsity=config.mask_rate, sparsity_distribution=config.sparsity_distribution)
    initial_global_params = copy.deepcopy(net_glob.state_dict())

    # initialize clients
    clients = {}
    client_ids = []
    config.proportion = [1, 1, 1, 1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(args.nusers):
        cl = Client_PruneFL(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                            dataset_test[client_id],
                            local_net=all_models[args.dataset](config, device))
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    # pick a client randomly and perform some local readjustment there only.
    # all clients are equally bad in this setting
    rng = np.random.default_rng()
    initial_client = clients[rng.choice(list(clients.keys()), size=1)[0]]
    last_differences = []
    for r in tqdm(range(args.initial_rounds)):
        global_params = net_glob.state_dict()
        readjust = True
        train_result = initial_client.train(w_server=global_params, round=r)

        gradients = []
        for name, param in initial_client.local_net.named_parameters():
            if not needs_mask(name, mask_weight_indicator):
                continue
            gradients.append(param.grad)

        if readjust:
            diff = initial_client.local_net.prunefl_readjust(gradients, layer_times, prunable_params=0.3)
            last_differences.append(diff)
        net_glob.load_state_dict(initial_client.local_net.state_dict())
        if len(last_differences) >= 5 and all(x < 0.1 for x in last_differences[-5:]):
            break

    for round in range(args.rounds):
        client_capacity = make_capacity(config, client_capacity)
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []
        w_locals, loss_locals, acc_locals = [], [], []

        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client

        w_glob = net_glob.state_dict()
        aggregated_params = {}
        aggregated_params_for_mask = {}
        aggregated_masks = {}
        w_locals = []
        aggregated_gradients = []
        # set server parameters to 0 in preparation for aggregation,
        for name, param in w_glob.items():
            if name.endswith('_mask'):
                continue
            aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
            aggregated_params_for_mask[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
            if needs_mask(name, mask_weight_indicator):
                aggregated_masks[name] = torch.zeros_like(param, device='cpu')
                aggregated_gradients.append(torch.zeros_like(param, device='cpu'))

        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            readjust = (round - 1) % config.rounds_between_readjustments == 0

            # 进行本地训练
            train_result = client.train(w_server=w_glob, round=round)
            cl_params = train_result['state']
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

            # first deduce masks for the received weights
            cl_weight_params = {}
            cl_mask_params = {}
            for name, cl_param in cl_params.items():
                if name.endswith('_orig'):
                    name = name[:-5]
                elif name.endswith('_mask'):
                    name = name[:-5]
                    cl_mask_params[name] = cl_param.to(device='cpu', copy=True)
                    continue

                cl_weight_params[name] = cl_param.to(device='cpu', copy=True)

            # at this point, we have weights and masks (possibly all-ones)
            # for this client. we will proceed by applying the mask and adding
            # the masked received weights to the aggregate, and adding the mask
            # to the aggregate as well.
            for name, cl_param in cl_weight_params.items():
                if name in cl_mask_params:
                    # things like weights have masks
                    cl_mask = cl_mask_params[name]
                    sv_mask = w_glob[name + '_mask'].to('cpu', copy=True)

                    aggregated_params[name].add_(client.train_size() * cl_param * cl_mask)
                    aggregated_params_for_mask[name].add_(client.train_size() * cl_param * cl_mask)
                    aggregated_masks[name].add_(client.train_size() * cl_mask)
                else:
                    # things like biases don't have masks
                    aggregated_params[name].add_(client.train_size() * cl_param)

            # get gradients
            grad_i = 0
            for name, param in client.local_net.named_parameters():
                if not needs_mask(name, mask_weight_indicator):
                    continue
                aggregated_gradients[grad_i].add_(param.grad.to('cpu'))
                grad_i += 1

        # divide gradients
        for g in aggregated_gradients:
            g.div_(len(idxs_users))

        # at this point, we have the sum of client parameters
        # in aggregated_params, and the sum of masks in aggregated_masks. We
        # can take the average now by simply dividing...
        for name, param in aggregated_params.items():

            # if this parameter has no associated mask, simply take the average.
            if name not in aggregated_masks:
                aggregated_params[name] /= sum(clients[i].train_size() for i in idxs_users)
                # aggregated_params_for_mask[name] /= sum(clients[i].train_size() for i in client_indices)
                continue

            # otherwise, we are taking the weighted average w.r.t. the number of
            # samples present on each of the clients.
            aggregated_params[name] /= aggregated_masks[name]
            aggregated_params_for_mask[name] /= aggregated_masks[name]
            aggregated_masks[name] /= aggregated_masks[name]

            # it's possible that some weights were pruned by all clients. In this
            # case, we will have divided by zero. Those values have already been
            # pruned out, so the values here are only placeholders.
            aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                       nan=0.0, posinf=0.0, neginf=0.0)
            aggregated_params_for_mask[name] = torch.nan_to_num(aggregated_params[name],
                                                                nan=0.0, posinf=0.0, neginf=0.0)
            aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                      nan=0.0, posinf=0.0, neginf=0.0)

        aggregated_params_for_mask = average_w(w_locals)

        # masks are parameters too!
        for name, mask in aggregated_masks.items():
            aggregated_params[name + '_mask'] = mask
            aggregated_params_for_mask[name + '_mask'] = mask

        # reset global params to aggregated values
        net_glob.load_state_dict(aggregated_params_for_mask)

        # evaluate performance
        torch.cuda.empty_cache()
        train_loss, train_acc = [], []
        glob_val_loss, glob_val_acc = [], []
        glob_test_loss, glob_test_acc = [], []
        for _, c in clients.items():
            train_res = evaluate(config, c.traindata_loader, copy.deepcopy(net_glob), device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            glob_val_res = evaluate(config, c.valdata_loader, copy.deepcopy(net_glob), device)
            glob_val_loss.append(glob_val_res[0])
            glob_val_acc.append(glob_val_res[1])
            glob_test_res = evaluate(config, c.testdata_loader, copy.deepcopy(net_glob), device)
            glob_test_loss.append(glob_test_res[0])
            glob_test_acc.append(glob_test_res[1])
        print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, sum(train_loss) / len(train_loss),
                                                                              sum(train_acc) / len(train_acc)))
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)), flush=True)
        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(sum(glob_val_loss) / len(glob_val_loss),
                                            sum(glob_val_acc) / len(glob_val_acc)))
        print("test loss: {:.5f}, test accuracy: {:.5f}".format(sum(glob_test_loss) / len(glob_test_loss),
                                                                sum(glob_test_acc) / len(glob_test_acc)), flush=True)

        if readjust:
            net_glob.prunefl_readjust(aggregated_gradients, layer_times, prunable_params=0.3 * 0.5 ** round)
