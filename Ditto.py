#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os

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
import warnings
from Client import Client, evaluate
from FedAvg import make_capacity

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]


class Client_Ditto(Client):

    def update_weights_ditto(self, weights, local_ep, round, w_glob=None, personalized=False, lr=None):
        '''Train the client network for a single round.'''

        epoch_losses, epoch_acc = [], []
        if lr:
            self.learning_rate = lr
        self.local_net.load_state_dict(copy.deepcopy(weights))
        self.local_net = self.local_net.to(self.device)
        dl_cost = 0
        for key in weights:
            dl_cost += torch.numel(weights[key]) * 32
        self.local_net.train()
        self.reset_optimizer(round=round)

        for iter in range(local_ep):
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
                if personalized:
                    reg = 0.0
                    for param_index, param in enumerate(self.local_net.parameters()):
                        reg += (self.args.lamda * torch.norm(param - list(w_glob.values())[param_index]))
                    loss += reg
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
        train_flops = len(self.traindata_loader) * count_training_flops(self.local_net,
                                                                        self.args) * self.args.local_ep

        ret = dict(state=self.local_net.state_dict(),
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
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

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_Ditto_{}.pt'.format(args.dataset, args.seed)

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
        cl = Client_Ditto(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
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
        w_locals, loss_locals, avg_weight, acc_locals = [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            avg_weight.append(data_num[idx])
            train_result = client.update_weights(copy.deepcopy(w_glob), round)
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            per_train_result = client.update_weights_ditto(weights=copy.deepcopy(client.model_trans.state_dict()),
                                                           local_ep=config.local_ep, round=round, personalized=True,
                                                           w_glob=copy.deepcopy(w_glob))
            client.model_trans.load_state_dict(per_train_result['state'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

        w_glob = avg(w_locals, w_glob, args, device)
        net_glob.load_state_dict(w_glob)

        # train_loss, train_acc = [], []
        # val_loss, val_acc = [], []
        # test_loss, test_acc = [], []
        # for _, c in clients.items():
        #     train_res = evaluate(config, c.traindata_loader, net_glob, device)
        #     train_loss.append(train_res[0])
        #     train_acc.append(train_res[1])
        #     val_res = evaluate(config, c.valdata_loader, net_glob, device)
        #     val_loss.append(val_res[0])
        #     val_acc.append(val_res[1])
        #     test_res = evaluate(config, c.testdata_loader, net_glob, device)
        #     test_loss.append(test_res[0])
        #     test_acc.append(test_res[1])
        # print('\nTrain loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
        #                                                             sum(train_acc) / len(train_acc)))
        # print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
        #       'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        # print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        # print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        # print('Max local training time: {:.5f} s'.format(max(training_time)))

        # print("Global Round {}, Validation loss: {:.5f}, "
        #       "val accuracy: {:.5f}".format(round, sum(val_loss) / len(val_loss),
        #                                     sum(val_acc) / len(val_acc)))
        # print("Global Round {}, test loss: {:.5f}, "
        #       "test accuracy: {:.5f}".format(round, sum(test_loss) / len(test_loss),
        #                                      sum(test_acc) / len(test_acc)), flush=True)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []
        for _, c in clients.items():
            train_res = evaluate(config, c.traindata_loader, c.model_trans, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            val_res = evaluate(config, c.valdata_loader, c.model_trans, device)
            val_loss.append(val_res[0])
            val_acc.append(val_res[1])
            test_res = evaluate(config, c.testdata_loader, c.model_trans, device)
            test_loss.append(test_res[0])
            test_acc.append(test_res[1])
        print('\nPersonalized Train loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                                 sum(train_acc) / len(train_acc)))

        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)))

        print("Validation loss: {:.5f}, val accuracy: {:.5f}".format(sum(val_loss) / len(val_loss),
                                                                     sum(val_acc) / len(val_acc)))
        print("test loss: {:.5f}, test accuracy: {:.5f}".format(sum(test_loss) / len(test_loss),
                                                                sum(test_acc) / len(test_acc)), flush=True)