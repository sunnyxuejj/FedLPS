import os
import math
import random
import pickle

from Client import Client, evaluate
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image, repackage_hidden
from Models import all_models
from data.reddit.user_data import data_process
from FedLPS import make_model_rate
from utils.main_flops_counter import count_training_flops
import warnings

warnings.filterwarnings('ignore')

args = args_parser()
FLOPS_capacity = [46528.0, 93056.0, 186112.0, 372224.0, 744448.0]
# FLOPS_capacity = [186112.0, 372224.0, 744448.0]
# FLOPS_capacity = [372224.0, 744448.0]


def mask_weight_magnitude(args, weight, mask_rate=0.5):
    weight_magnitude = torch.sum(torch.abs(weight), list(range(1, weight.dim())))
    topk_values, _ = torch.topk(weight_magnitude.cpu().detach().flatten(),
                                max(int(mask_rate * weight_magnitude.size(0)), 1))
    threshold = np.min(topk_values.cpu().detach().numpy())
    new_mask = torch.where(weight_magnitude < threshold, 0, 1).view(-1, 1)
    for j in range(2, weight.dim()):
        new_mask = new_mask.unsqueeze(dim=1)
    mask_weight = new_mask.expand_as(weight).to(args.device)
    return weight * mask_weight.float(), mask_weight.float()


class Client_FedMP(Client):
    def __init__(self, *args, **kwargs):
        super(Client_FedMP, self).__init__(*args, **kwargs)
        a = np.linspace(0.2, 1, 2)
        self.mab_arms = [[a[i], a[i + 1]] for i in range(len(a) - 1)]
        self.arms_reward = {}
        for i in range(len(self.mab_arms)):
            self.arms_reward[self.mab_arms[i][0]] = [0.0]
        self.pull_times = np.ones(len(self.mab_arms))
        self.loss_record = [10]
        self.selected_times = 0
    def E_UCB(self):
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

    def update_weights_MP(self, w_server, round, lr=None):
        '''Train the client network for a single round.'''

        self.selected_times += 1
        epoch_losses, epoch_acc = [], []
        if lr:
            self.learning_rate = lr
        global_weight_collector = []
        for name in w_server:
            global_weight_collector.append(copy.deepcopy(w_server[name]).to(self.args.device))
        w_local_ini = copy.deepcopy(w_server)
        self.local_net.load_state_dict(copy.deepcopy(w_local_ini))
        self.local_net = self.local_net.to(self.device)
        dl_cost = self.local_net.param_size
        self.local_net.train()


        for name, param in self.local_net.named_parameters():
            if name in self.local_net.mask_weight_indicator:
                if self.args.dataset == 'reddit' and 'encoder' in name:
                    param.data, self.local_net.mask_bool_layer_dict[name] = mask_weight_magnitude(self.args,
                                                                                                  param.data, 0.8)
                else:
                    param.data, self.local_net.mask_bool_layer_dict[name] = mask_weight_magnitude(self.args,
                                                                                                  param.data,
                                                                                                  self.mask_rate)

        self.reset_optimizer(round=round)

        for iter in range(self.args.local_ep):
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
                        fed_prox_reg += ((self.args.lamda / 2) * torch.norm(
                            (param - global_weight_collector[param_index])) ** 2)
                    loss += fed_prox_reg
                loss.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.args.clip)

                for name, param in self.local_net.named_parameters():
                    if name in self.local_net.mask_weight_indicator:
                        param.grad.data = param.grad.data * self.local_net.mask_bool_layer_dict[name].to(self.device)

                self.optimizer.step()
                list_loss.append(loss.item())
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                corrent += torch.sum(torch.eq(pred_labels, y)).item()

            acc = corrent / total
            epoch_acc.append(acc)
            epoch_losses.append(sum(list_loss) / len(list_loss))

        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        ul_cost = self.local_net.stat_param_sizes()
        cur_params = copy.deepcopy(self.local_net.state_dict())
        train_flops = len(self.traindata_loader) * count_training_flops(copy.deepcopy(self.local_net),
                                                                        self.args) * self.args.local_ep
        self.loss_record.append(sum(epoch_losses) / len(epoch_losses))

        grad = {}
        for k in cur_params:
            if self.args.mask:
                if k in self.local_net.mask_weight_name:
                    continue
                if k in self.local_net.mask_weight_indicator:
                    cur_params[k] = cur_params[k] * self.local_net.mask_bool_layer_dict[k].clone().detach()
                    w_server[k] = w_server[k] * self.local_net.mask_bool_layer_dict[k].clone().detach()
            grad[k] = cur_params[k] - w_server[k]

        ret = dict(state=grad,
                   loss=sum(epoch_losses) / len(epoch_losses),
                   acc=sum(epoch_acc) / len(epoch_acc),
                   dl_cost=dl_cost, ul_cost=ul_cost,
                   train_flops=train_flops)
        return ret


if __name__ == "__main__":
    config = args
    config.mask = True
    config.mask_magnitude = True
    args.FedMP_online = True

    torch.cuda.set_device(config.gpu)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.gpu != -1:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    config.device = device

    dataset_train, dataset_val, dataset_test, data_num = {}, {}, {}, {}
    if config.dataset == 'mnist':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_mnist_extr_noniid(
            config.nusers,
            config.nclass,
            config.nsamples,
            config.rate_unbalance)
        for i in range(config.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif config.dataset == 'cifar10':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar10_extr_noniid(
            config.nusers,
            config.nclass,
            config.nsamples,
            config.rate_unbalance)
        for i in range(config.nusers):
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
    elif config.dataset == 'reddit':
        # dataload
        data_dir = 'data/reddit/train/'
        with open('data/reddit/reddit_vocab.pck', 'rb') as f:
            vocab = pickle.load(f)
        nvocab = vocab['size']
        config.nvocab = nvocab
        train_data, val_data, test_data = data_process(data_dir, nvocab, config.nusers)
        for i in range(config.nusers):
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
    os.makedirs(f'./log/{config.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_FedMP_{}.pt'.format(config.dataset, config.seed)

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
    if last_layer in mask_weight_indicator:
        mask_weight_indicator = mask_weight_indicator[:-1]
    config.mask_weight_indicator = copy.deepcopy(mask_weight_indicator)

    # initialized global model
    net_glob = all_models[config.dataset](config, device)
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
    client_capacity = np.array(FLOPS_capacity)[rate_idx]
    for client_id in range(config.nusers):
        cl = Client_FedMP(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                          dataset_test[client_id],
                          local_net=all_models[config.dataset](config, device), mask_rate=clients_mask_rate[client_id])
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    for round in range(config.rounds):
        clients_mask_rate, client_capacity = make_model_rate(config, clients_mask_rate, client_capacity)
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []

        net_glob.train()
        w_locals, loss_locals, acc_locals = [], [], []
        m = max(int(config.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            if args.FedMP_online:
                client.mask_rate, action_index = client.E_UCB()
            train_result = client.update_weights_MP(w_server=copy.deepcopy(w_glob), round=round)
            w_locals.append(train_result['state'])
            loss_locals.append(train_result['loss'])
            acc_locals.append(train_result['acc'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])
            training_time.append(train_result['train_flops'] / 1e6 / client_capacity[idx])

        w_glob = avg(w_locals, w_glob, config, device)
        net_glob.load_state_dict(w_glob)
        net_glob = net_glob.to(device)

        # E_UCB
        i = 0
        for idx in idxs_users:
            client = clients[idx]
            _, action_index = client.E_UCB()
            reward = (client.loss_record[-1] - client.loss_record[-2]) / \
                     (training_time[i] - (sum(training_time) / len(training_time)))
            client.arms_reward[client.mab_arms[action_index][0]].append(reward.item())
            i += 1

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        glob_test_loss, glob_test_acc = [], []
        for _, c in clients.items():
            train_res = evaluate(config, c.traindata_loader, net_glob, device)
            train_loss.append(train_res[0])
            train_acc.append(train_res[1])
            glob_val_res = evaluate(config, c.valdata_loader, net_glob, device)
            val_loss.append(glob_val_res[0])
            val_acc.append(glob_val_res[1])
            glob_test_res = evaluate(config, c.testdata_loader, copy.deepcopy(net_glob), device)
            glob_test_loss.append(glob_test_res[0])
            glob_test_acc.append(glob_test_res[1])
        train_loss, train_acc = np.array(train_loss)[~np.isnan(np.array(train_loss))], np.array(train_acc)[
            ~np.isnan(np.array(train_acc))]
        val_loss, val_acc = np.array(val_loss)[~np.isnan(np.array(val_loss))], np.array(val_acc)[
            ~np.isnan(np.array(val_acc))]
        glob_test_loss, glob_test_acc = np.array(glob_test_loss)[~np.isnan(np.array(glob_test_loss))], \
        np.array(glob_test_acc)[
            ~np.isnan(np.array(glob_test_acc))]
        print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, np.mean(train_loss),
                                                                              np.mean(train_acc)))
        print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
              'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
        print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
        print('Max time for upload time: {:.5f} s, Max time for download: {:.5f} s'.format(max(uptime), max(down_time)))
        print('Max local training time: {:.5f} s'.format(max(training_time)), flush=True)
        print("Validation loss: {:.5f}, "
              "val accuracy: {:.5f}".format(np.mean(val_loss), np.mean(val_acc)), flush=True)
        print("test loss: {:.5f}, "
              "test accuracy: {:.5f}".format(np.mean(glob_test_loss), np.mean(glob_test_acc)), flush=True)
