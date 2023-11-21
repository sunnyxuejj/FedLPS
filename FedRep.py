#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os
import pickle
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, train_val_test_image, repackage_hidden
from utils.main_flops_counter import count_training_flops
from Models import all_models
from data.reddit.user_data import data_process
import warnings
from Client import Client, evaluate
warnings.filterwarnings('ignore')

args = args_parser()


class Client_Rep(Client):

    def update_weights_rep(self, weights, round, lr=None):
        '''Train the client network for a single round.'''

        epoch_losses, epoch_acc = [], []
        if lr:
            self.learning_rate = lr
        for key in self.args.personal_layers:
            weights[key] = copy.deepcopy(self.local_net.state_dict()[key])
        self.local_net.load_state_dict(copy.deepcopy(weights))
        self.local_net = self.local_net.to(self.device)
        dl_cost = 0
        for key in weights:
            if key not in self.args.personal_layers:
                dl_cost += torch.numel(weights[key]) * 32
        self.local_net.train()
        self.reset_optimizer(round=round)

        for iter in range(self.args.client_all_epoch):
            if iter < self.args.personal_epoch:
                for name, param in self.local_net.named_parameters():
                    if name in self.args.personal_layers:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in self.local_net.named_parameters():
                    if name in self.args.personal_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
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
        cur_params = copy.deepcopy(self.local_net.state_dict())
        ul_cost, grad = 0, {}
        for key in cur_params:
            if key not in self.args.personal_layers:
                grad[key] = cur_params[key] - weights[key]
                ul_cost += torch.numel(cur_params[key]) * 32
        train_flops = len(self.traindata_loader) * count_training_flops(self.local_net,
                                                                        self.args) * self.args.client_all_epoch

        ret = dict(state=grad,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret


if __name__ == "__main__":
    config = args
    config.client_all_epoch = 8
    config.personal_epoch = 3

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
        all_val, all_test = [], []
        for value in val_data.values():
            for sent in value:
                all_val.append(sent)
        for value in test_data.values():
            for sent in value:
                all_test.append(sent)
    elif args.dataset == 'mnist':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_mnist_extr_noniid(args.nusers,
                                                                                                      args.nclass,
                                                                                                      args.nsamples,
                                                                                                      args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar10':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar10_extr_noniid(args.nusers,
                                                                                                      args.nclass,
                                                                                                      args.nsamples,
                                                                                                      args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])

    best_val_acc = None
    os.makedirs(f'./log/{args.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_FedRep_{}.pt'.format(args.dataset, args.seed)

    # 确定哪些层个性化
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
    personal_layers = []
    personal_layers.append(layers[-2])
    personal_layers.append(layers[-1])
    config.personal_layers = personal_layers

    # initialized global model
    net_glob = all_models[args.dataset](config, device)
    net_glob = net_glob.to(device)
    w_glob = net_glob.state_dict()
    for key in config.personal_layers:
        del w_glob[key]
    # initialize clients
    clients = {}
    client_ids = []
    for client_id in range(args.nusers):
        cl = Client_Rep(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
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
        upload_cost_round = []
        download_cost_round = []
        compute_flops_round = []

        net_glob.train()
        w_locals, loss_locals, avg_weight, acc_locals = [], [], [], []
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            avg_weight.append(data_num[idx])
            train_result = client.update_weights_rep(copy.deepcopy(w_glob), round)
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost[i] += train_result['dl_cost']
            download_cost_round.append(train_result['dl_cost'])
            upload_cost[i] += train_result['ul_cost']
            upload_cost_round.append(train_result['ul_cost'])
            compute_flops[i] += (train_result['train_flops'])
            compute_flops_round.append(train_result['train_flops'])

        w_glob = avg(w_locals, w_glob, args, device)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []

        local_model = copy.deepcopy(net_glob)
        local_params = copy.deepcopy(w_glob)
        for _, c in clients.items():
            for key in config.personal_layers:
                local_params[key] = c.local_net.state_dict()[key]
            local_model.load_state_dict(local_params)
            train_res = evaluate(config, c.traindata_loader, local_model, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            val_res = evaluate(config, c.valdata_loader, local_model, device)
            val_loss.append(val_res[0])
            val_acc.append(val_res[1])
            test_res = evaluate(config, c.testdata_loader, local_model, device)
            test_loss.append(test_res[0])
            test_acc.append(test_res[1])
        print('\nPersonalized Train loss: {:.5f}, train accuracy: {:.5f}'.format(sum(train_loss) / len(train_loss),
                                                                                 sum(train_acc) / len(train_acc)))
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print("Personalized Round {}, Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(round, sum(val_loss) / len(val_loss),
                                            sum(val_acc) / len(val_acc)))
        print("Personalized Round {}, test loss: {:.5f}, "
              "test accuracy: {:.5f}".format(round, sum(test_loss) / len(test_loss),
                                             sum(test_acc) / len(test_acc)))