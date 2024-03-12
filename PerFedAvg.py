#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os

import pickle
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image, repackage_hidden
from utils.main_flops_counter import count_training_flops
from Models import all_models
from data.reddit.user_data import data_process
import warnings
from Client import Client, evaluate
from FedAvg import make_capacity

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]


class Client_PerFedAvg(Client):

    def update_weights_perFedavg(self, weights, round, lr=None):
        '''Train the client network for a single round.'''

        epoch_losses, epoch_acc = [], []
        if lr:
            self.learning_rate = lr
        lr_in = self.learning_rate * 0.001
        self.local_net.load_state_dict(copy.deepcopy(weights))
        self.local_net = self.local_net.to(self.device)
        if self.args.dataset == 'reddit':
            hidden = self.local_net.init_hidden(self.args.local_bs)
            hidden_tar = self.local_net.init_hidden(self.args.local_bs)
        dl_cost = 0
        for key in weights:
            dl_cost += torch.numel(weights[key]) * 32
        self.local_net.train()
        self.reset_optimizer(round=round)

        for iter in range(self.args.local_ep):
            list_loss = []
            total, corrent = 0, 0
            for batch_ind, local_data in enumerate(self.traindata_loader):
                param_dict = dict()
                for name, param in self.local_net.named_parameters():
                    if param.requires_grad:
                        param_dict[name] = param.to(device=self.device)
                names_weights_copy = param_dict
                self.local_net.zero_grad()

                if self.args.dataset == 'reddit':
                    x = torch.stack(local_data[:-1]).to(self.device)
                    y = torch.stack(local_data[1:]).view(-1).to(self.device)
                    tar_x = torch.stack(local_data[:-1]).to(self.device)
                    tar_y = torch.stack(local_data[1:]).view(-1).to(self.device)
                    total += y.size(0)
                    hidden = repackage_hidden(hidden)
                    if hidden[0][0].size(1) != x.size(1):
                        hidden = self.local_net.init_hidden(x.size(1))
                    out, hidden = self.local_net(x, hidden)
                else:
                    x, y = local_data[0].to(self.device), local_data[1].to(self.device)
                    tar_x, tar_y = local_data[0].to(self.device), local_data[1].to(self.device)
                    total += len(y)
                    out = self.local_net(x)

                loss_sup = self.loss_func(out, y)
                list_loss.append(loss_sup.item())
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                corrent += torch.sum(torch.eq(pred_labels, y)).item()

                grads = torch.autograd.grad(loss_sup, names_weights_copy.values(), create_graph=True, allow_unused=True)
                names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
                for key, grad in names_grads_copy.items():
                    if grad is not None:
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                        names_weights_copy[key] = names_weights_copy[key] - lr_in * names_grads_copy[key]

                if self.args.dataset == 'reddit':
                    if hidden_tar[0][0].size(1) != tar_x.size(1):
                        hidden_tar = self.local_net.init_hidden(x.size(1))
                        hidden_tar = repackage_hidden(hidden_tar)
                    tar_out, hidden_tar = self.local_net(tar_x, hidden_tar)
                else:
                    tar_out = self.local_net(tar_x)
                    loss_tar = self.loss_func(tar_out, tar_y)
                    loss_tar.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.args.clip)
                self.optimizer.step()

                del tar_out.grad
                del loss_sup.grad
                del out.grad
                self.optimizer.zero_grad()
                self.local_net.zero_grad()

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        cur_params = copy.deepcopy(self.local_net.state_dict())
        ul_cost = 0
        for key in weights:
            ul_cost += torch.numel(weights[key]) * 32
        train_flops = 2 * len(self.traindata_loader) * count_training_flops(self.local_net,
                                                                            self.args) * self.args.local_ep

        ret = dict(state=cur_params,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret

    def local_finetune(self, model):
        '''Train the client network for a single round.'''
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=self.args.lr, weight_decay=self.args.wdecay)
        if self.args.dataset == 'reddit':
            hidden = model.init_hidden(self.args.local_bs)
            hidden = repackage_hidden(hidden)
        model.train()

        for iter in range(self.args.local_ep):
            for batch_ind, local_data in enumerate(self.traindata_loader):
                optimizer.zero_grad()

                if self.args.dataset == 'reddit':
                    x = torch.stack(local_data[:-1]).to(self.device)
                    y = torch.stack(local_data[1:]).view(-1).to(self.device)
                    if hidden[0][0].size(1) != x.size(1):
                        hidden = model.init_hidden(x.size(1))
                        hidden = repackage_hidden(hidden)
                    out, hidden = model(x, hidden)
                else:
                    x, y = local_data[0].to(self.device), local_data[1].to(self.device)
                    out = model(x)

                loss = self.loss_func(out, y)
                loss.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)

                optimizer.step()
        return model


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

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_PFA_{}.pt'.format(args.dataset, args.seed)

    # 确定哪些层个性化
    config.mask_weight_indicator = []
    config.personal_layers = []

    # initialized global model
    net_glob = all_models[args.dataset](config, device)
    net_glob = net_glob.to(device)
    w_glob = net_glob.state_dict()
    for key in config.personal_layers:
        del w_glob[key]
    # initialize clients
    clients = {}
    client_ids = []
    config.proportion = [1, 1, 1, 1, 1]
    rate_idx = torch.multinomial(torch.tensor(config.proportion).float(), num_samples=config.nusers,
                                 replacement=True).tolist()
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(args.nusers):
        cl = Client_PerFedAvg(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                              dataset_test[client_id],
                              local_net=all_models[args.dataset](config, device))
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    # we need to accumulate compute/DL/UL costs regardless of round number, resetting only
    # when we actually report these numbers
    compute_flops = np.zeros(len(clients))  # the total floating operations for each client
    download_cost = np.zeros(len(clients))
    upload_cost = np.zeros(len(clients))

    for round in range(args.rounds):
        client_capacity = make_capacity(config, client_capacity)
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []

        net_glob.train()
        w_locals, loss_locals, avg_weight, acc_locals = [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            avg_weight.append(data_num[idx])
            train_result = client.update_weights_perFedavg(copy.deepcopy(w_glob), round)
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

        w_glob = average_weights(w_locals, avg_weight, args)
        net_glob.load_state_dict(w_glob)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []
        for _, c in clients.items():
            train_res = evaluate(config, c.traindata_loader, net_glob, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            val_res = evaluate(config, c.valdata_loader, net_glob, device)
            val_loss.append(val_res[0])
            val_acc.append(val_res[1])
            test_res = evaluate(config, c.testdata_loader, net_glob, device)
            test_loss.append(test_res[0])
            test_acc.append(test_res[1])
        print('\nTrain loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                    sum(train_acc) / len(train_acc)), flush=True)
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)), flush=True)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []
        for _, c in clients.items():
            train_res = evaluate(config, c.traindata_loader, c.local_net, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            val_res = evaluate(config, c.valdata_loader, c.local_net, device)
            val_loss.append(val_res[0])
            val_acc.append(val_res[1])
            test_res = evaluate(config, c.testdata_loader, c.local_net, device)
            test_loss.append(test_res[0])
            test_acc.append(test_res[1])
        print("Validation loss: {:.5f}, val accuracy: {:.5f}".format(sum(val_loss) / len(val_loss),
                                                                     sum(val_acc) / len(val_acc)))
        print("test loss: {:.5f}, test accuracy: {:.5f}".format(sum(test_loss) / len(test_loss),
                                                                sum(test_acc) / len(test_acc)), flush=True)

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    test_loss, test_acc = [], []
    for _, c in clients.items():
        local_model = c.local_finetune(copy.deepcopy(net_glob))
        train_res = evaluate(config, c.traindata_loader, local_model, device)
        train_loss.append(train_res[0])
        train_acc.append(train_res[1])
        val_res = evaluate(config, c.valdata_loader, local_model, device)
        val_loss.append(val_res[0])
        val_acc.append(val_res[1])
        test_res = evaluate(config, c.testdata_loader, local_model, device)
        test_loss.append(test_res[0])
        test_acc.append(test_res[1])
    print('\nLast Personalized Train loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                                  sum(train_acc) / len(train_acc)))
    print("Last Personalized Round {}, Validation loss: {:.5f}, "
          "val accuracy: {:.5f}".format(round, sum(val_loss) / len(val_loss),
                                        sum(val_acc) / len(val_acc)))
    print("Last Personalized Round {}, test loss: {:.5f}, "
          "test accuracy: {:.5f}".format(round, sum(test_loss) / len(test_loss),
                                         sum(test_acc) / len(test_acc)), flush=True)
    