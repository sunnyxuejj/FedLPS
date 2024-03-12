import copy
import torch
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
        self.learning_rate = args.lr

        self.local_epochs = args.local_ep

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.local_net = local_net.to(device)
        self.model_trans = copy.deepcopy(self.local_net)
        self.mask_rate = mask_rate

        if self.args.dataset == 'reddit' or self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100' or self.args.dataset == 'tinyimagenet':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss_func = nn.NLLLoss().to(self.device)
        self.reset_optimizer()

        self.train_data = train_data
        self.traindata_loader = DataLoader(train_data, batch_size=self.args.local_bs, shuffle=True)
        self.valdata_loader = DataLoader(val_data, batch_size=self.args.local_bs, shuffle=False)
        self.testdata_loader = DataLoader(test_data, batch_size=self.args.local_bs, shuffle=False)

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

        epoch_losses, epoch_acc = [], []

        if lr:
            self.learning_rate = lr
        global_weight_collector = []
        for name in w_server:
            global_weight_collector.append(copy.deepcopy(w_server[name]).to(self.args.device))
        self.local_net.load_state_dict(copy.deepcopy(w_server))
        self.local_net = self.local_net.to(self.device)
        dl_cost = 0
        for key in w_server:
            dl_cost += torch.numel(w_server[key]) * 32
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
                if self.args.fedprox:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(self.local_net.parameters()):
                        fed_prox_reg += ((self.args.lamda / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss += fed_prox_reg
                loss.backward()
                for name, params in self.local_net.named_parameters():
                    a = params.grad.data
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
        self.model_trans.load_state_dict(copy.deepcopy(cur_params))
        train_flops = len(self.traindata_loader) * count_training_flops(self.model_trans, self.args) * self.args.local_ep

        grad, ul_cost = {}, 0
        for k in cur_params:
            grad[k] = cur_params[k] - w_server[k]
            ul_cost += torch.numel(cur_params[k]) * 32

        ret = dict(state=cur_params,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret
