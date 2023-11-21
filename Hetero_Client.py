import copy
import math
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from util import repackage_hidden
from utils.main_flops_counter import count_training_flops


def evaluate(args, data_loader, model, device):
    """ Loss and accuracy of the given data with the given model. """

    model = model.to(device)
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    if args.dataset == 'reddit':
        hidden = model.init_hidden(args.local_bs)
    with torch.no_grad():
        total, corrent = 0, 0
        loss_list = []
        for val_idx, val_data in enumerate(data_loader):
            if args.dataset == 'reddit':
                x = torch.stack(val_data[:-1]).to(device)
                y = torch.stack(val_data[1:]).view(-1).to(device)
                if hidden[0][0].size(1) != x.size(1):
                    hidden = model.init_hidden(x.size(1))
                    out, hidden = model(x, hidden)
                out, hidden = model(x, hidden)
                total += y.size(0)
            else:
                x, y = val_data[0].to(device), val_data[1].to(device)
                total += len(y)
                out = model(x)
            loss = loss_func(out, y)
            loss_list.append(loss.item())
            _, pred_labels = torch.max(out, 1)
            pred_labels = pred_labels.view(-1)
            corrent += torch.sum(torch.eq(pred_labels, y)).item()
        eval_acc = corrent / total
        loss_avg = sum(loss_list) / len(loss_list)
    return [loss_avg, eval_acc]


class Hetero_Client:

    def __init__(self, args, device, id, train_data, val_data, test_data, local_net, mask_rate):
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
        local_net:
        mask_rate:

        Returns: a new client.
        '''

        self.args = args
        self.device = device
        self.id = id
        self.selected_times = 0

        self.learning_rate = args.lr
        self.local_epochs = args.local_ep

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.local_net = local_net.to(device)
        self.model_trans = copy.deepcopy(self.local_net)

        a = np.linspace(0.5, 1, 6)
        self.mab_arms = [[a[i], a[i+1]] for i in range(len(a) - 1)]
        self.arms_reward = {}
        for i in range(len(self.mab_arms)):
            self.arms_reward[self.mab_arms[i][0]] = [0.0]
        self.pull_times = np.ones(len(self.mab_arms))

        self.mask_rate = mask_rate
        self.rec_mask_rate = None
        self.best_val_acc = None
        self.final_parameters = None
        self.acc_record = [0]

        layers = list(self.local_net.state_dict().keys())
        layers_name = []
        for key in layers:
            if 'weight' in key:
                layers_name.append(key)
        self.first_layer = layers_name[0]
        self.last_layer = layers_name[-1]

        if self.args.dataset == 'reddit' or self.args.dataset == 'cifar10':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        elif self.args.dataset == 'mnist':
            self.loss_func = nn.NLLLoss().to(self.device)
        self.l1_penalty = nn.L1Loss().to(self.device)
        self.l2_penalty = nn.MSELoss().to(self.device)
        self.reset_optimizer()

        self.traindata_loader = DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True)
        self.valdata_loader = DataLoader(val_data, batch_size=self.args.local_bs, shuffle=True)
        self.testdata_loader = DataLoader(test_data, batch_size=self.args.local_bs, shuffle=False)

    def P_UCBV(self):
        T = self.args.rounds / (self.args.nusers * self.args.frac)
        phi = T / math.pow(len(self.mab_arms), 2)
        if self.selected_times == 0:
            self.es_0 = 1
            self.m = 0
            self.ep = 1
        score_list = []
        for j, (key, record) in enumerate(self.arms_reward.items()):
            r_ = np.mean(np.array(record))
            v_ = np.var(np.array(record))
            if math.log(phi * T * self.es_0) > 0:
                score = r_ + math.sqrt(0.5 * (abs(v_) + 2) * math.log(phi * T * self.es_0) / (4 * self.pull_times[j]))
            else:
                score = r_
            score_list.append(score)
        flag = random.random()
        if flag < self.ep:
            max_index = random.randint(0, len(self.mab_arms) - 1)
        else:
            _, max_index = np.max(np.array(score_list)), np.argmax(np.array(score_list))
        action = random.uniform(self.mab_arms[max_index][0], self.mab_arms[max_index][1])
        bound = self.mab_arms[max_index][1]
        self.mab_arms[max_index][1] = action
        self.mab_arms = np.insert(self.mab_arms, max_index + 1, [action, bound], axis=0)
        self.arms_reward[action] = copy.deepcopy(self.arms_reward[self.mab_arms[max_index][0]])
        pulltime_copy = self.pull_times[max_index]
        self.pull_times = np.insert(self.pull_times, max_index, pulltime_copy)
        self.pull_times[max_index + 1] += 1

        self.es_0 = self.es_0 / 2
        self.m += 1
        self.ep = 1 / self.m

        return action, max_index + 1

    def reset_optimizer(self, round=0):
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.local_net.parameters()),
                                             lr=self.args.lr * (self.args.lr_decay ** round),
                                             momentum=self.args.momentum, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_net.parameters()),
                                              lr=self.learning_rate, weight_decay=self.args.wdecay)

    def loss(self, out, y):
        sigm = nn.Sigmoid()
        weight_decay, sparseness = 0, 0
        loss = self.loss_func(out, y)
        for name, param in self.local_net.named_parameters():
            if name in self.local_net.mask_weight_indicator:
                weight_decay += self.args.wd * self.l2_penalty(
                    self.local_net.mask_layer_dict[name.replace('.', '_')].w,
                    sigm(torch.sum(torch.abs(param.data), list(range(1, param.data.dim())))))
        loss += weight_decay
        return loss

    def update_weights_leanable(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''
        epoch_losses, epoch_acc = [], []
        self.selected_times += 1
        global_weight_collector = copy.deepcopy(w_server)

        if lr:
            self.learning_rate = lr
        w_local_ini = copy.deepcopy(w_server)
        if self.args.mask:
            self.local_net.initialize_mask()
            self.model_trans.initialize_mask()
            w_local = copy.deepcopy(self.local_net.state_dict())
            local_params = copy.deepcopy(w_local_ini)
            for key in self.local_net.mask_weight_name:
                local_params[key] = w_local[key]
            self.local_net.load_state_dict(copy.deepcopy(local_params))
        else:
            self.local_net.load_state_dict(copy.deepcopy(w_local_ini))
        dl_cost = self.local_net.param_size
        self.local_net = self.local_net.to(self.device)
        self.local_net.train()

        self.reset_optimizer(round=round)

        for iter in range(self.local_epochs):
            list_loss = []
            total, corrent = 0, 0
            for batch_ind, local_data in enumerate(self.traindata_loader):
                self.optimizer.zero_grad()

                if self.local_net.training and self.args.mask:
                    i = 0
                    for name, param in self.local_net.named_parameters():
                        if name in self.local_net.mask_weight_indicator:
                            param.data, self.local_net.bias_layer_dict[name], self.local_net.mask_bool_layer_dict[
                                name], self.local_net.mask_label[self.local_net.mask_weight_name[i]] = \
                            self.local_net.mask_layer_dict[name.replace('.', '_')](param.data, self.mask_rate)
                            i = i + 1

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

                loss = self.loss(out, y)
                if self.args.fedprox:
                    fed_prox_reg = 0.0
                    for name, param in self.local_net.named_parameters():
                        if name in global_weight_collector:
                            fed_prox_reg += ((self.args.lamda / 2) * torch.norm(
                                (param - global_weight_collector[name])) ** 2)
                    loss += fed_prox_reg
                loss.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.args.clip)

                for name, param in self.local_net.named_parameters():
                    if name in self.local_net.mask_weight_indicator:
                        param.grad.data = param.grad.data * self.local_net.mask_bool_layer_dict[name].to(self.device)
                    elif name in self.local_net.mask_weight_name:
                        param.grad.data = param.grad.data * self.local_net.mask_label[name].to(self.device)

                self.optimizer.step()
                cur_params = copy.deepcopy(self.local_net.state_dict())
                self.local_net.weight_recover()
                list_loss.append(loss.item())
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                corrent += torch.sum(torch.eq(pred_labels, y)).item()

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

            local_model = copy.deepcopy(self.model_trans)
            local_model.load_state_dict(cur_params)
            val_res = evaluate(self.args, self.valdata_loader, local_model, self.device)
            if not self.best_val_acc or val_res[1] > self.best_val_loss:
                if self.mask_rate > self.args.mask_rate_list[2]:
                    self.rec_mask_rate = self.mask_rate
                    self.best_val_loss = val_res[1]
                    self.final_parameters = copy.deepcopy(cur_params)
                    for key in self.local_net.mask_weight_name:
                        del self.final_parameters[key]

        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.model_trans.load_state_dict(cur_params)
        ul_cost = self.model_trans.stat_param_sizes()
        train_flops = self.local_epochs * len(self.traindata_loader) * count_training_flops(self.model_trans, self.args)

        grad = {}
        for k in cur_params:
            if self.args.mask:
                if k in self.local_net.mask_weight_name:
                    continue
                if k in self.local_net.mask_weight_indicator:
                    cur_params[k] = cur_params[k] * self.local_net.mask_bool_layer_dict[k].clone().detach()
                    w_server[k] = w_server[k] * self.local_net.mask_bool_layer_dict[k].clone().detach()
            grad[k] = cur_params[k] - w_server[k]
        if self.args.dataset == 'reddit' and self.args.tie_weights:
            del grad[self.last_layer]

        train_acc = sum(epoch_acc) / len(epoch_acc)
        self.acc_record.append(train_acc)

        ret = dict(state=grad,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=train_acc,
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret
