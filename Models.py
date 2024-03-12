# -*- coding: utf-8 -*-
# @python: 3.8
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def needs_mask(name, mask_indicator):
    return name in mask_indicator


def mask_indicator(net, dtype=torch.float):
    childrens = list(net.children())
    if not childrens:
        for name, param in net.named_parameters():
            if 'weight' in name and not name.endswith('_mask'):
                if isinstance(net, torch.nn.Conv2d):
                    net.register_buffer(name + '_mask', torch.randn(param.size(0), dtype=dtype))
                if isinstance(net, torch.nn.Linear):
                    net.register_buffer(name + '_mask', torch.randn(param.size(0), dtype=dtype))
                if isinstance(net, torch.nn.LSTM):
                    net.register_buffer(name + '_mask', torch.randn(param.size(0), dtype=dtype))
                if isinstance(net, torch.nn.GRU):
                    net.register_buffer(name + '_mask', torch.randn(param.size(0), dtype=dtype))
                if isinstance(net, torch.nn.Embedding):
                    net.register_buffer(name + '_mask', torch.randn(param.size(0), dtype=dtype))
        return
    for c in childrens:
        mask_indicator(c)


def layer_flatten(net):
    childrens = list(net.children())
    if not childrens:
        if isinstance(net, torch.nn.LSTM):
            net.flatten_parameters()
        return
    for c in childrens:
        layer_flatten(c)


class WeightLayer(nn.Module):
    def __init__(self, input_size, device):
        super(WeightLayer, self).__init__()
        weight, _ = torch.sort(torch.randn(input_size, 1), 0, descending=True)
        self.w = nn.Parameter(weight.requires_grad_())
        self.device = device
        self.sigm = nn.Sigmoid()

    def forward(self, weights, mask_rate=0.5):
        topk_values, top_idx = torch.topk(self.sigm(self.w.clone().detach().flatten()),
                                          max(int(mask_rate * self.w.size(0)), 1))
        threshold = np.min(topk_values.cpu().detach().numpy())
        m = nn.Threshold(threshold, 0)
        mask = m(self.sigm(self.w))
        mask = torch.heaviside(mask, torch.tensor([0.0]).to(self.device))
        mask_label = mask.clone()
        for j in range(2, weights.dim()):
            mask = mask.unsqueeze(dim=1)
        mask_weight = mask.expand_as(weights).to(self.device)
        bias = weights - weights * mask_weight
        return weights * mask_weight.float(), bias.float(), mask_weight.float(), mask_label.float()

    def tailor(self, layer_params):
        mask = self.sigm(self.w)
        for j in range(2, layer_params.dim()):
            mask = mask.unsqueeze(dim=1)
        mask_weight = mask.expand_as(layer_params).to(self.device)
        return layer_params * mask_weight.float()


class CommenNet(nn.Module):
    '''Common functionality for all networks in this experiment.'''

    def __init__(self, config, device, mask_rate=1):
        super(CommenNet, self).__init__()
        self.args = config
        self.device = device
        self.communication_sparsity = 0

        # dropout-related parameters
        self.mask_rate = mask_rate
        self.mask_weight_indicator = config.mask_weight_indicator
        self.mask_layer_dict = nn.ModuleDict()
        self.mask_weight_name = []
        self.mask_bool_layer_dict = {}
        self.bias_layer_dict = {}
        self.mask_label = {}

    def init_param_sizes(self):
        # bits required to transmit mask and parameters?
        self.mask_size = 0
        self.param_size = 0
        for name, param in self.named_parameters():
            if name in self.mask_weight_name:
                continue
            param_size = torch.numel(param.data)
            self.param_size += param_size * 32  # FIXME: param.dtype.size?
            if name in self.mask_weight_indicator:
                self.mask_size += param_size

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def label_mask_weight(self):
        '''
        indicate the weight name that can be masked
        '''

        mask_indicator(self)
        for name, _ in self.state_dict().items():
            if name.endswith('_mask'):
                self.mask_weight_indicator.append(name[:-5])

    def initialize_mask(self):
        self.mask_weight_name = []
        for name, param in self.named_parameters():
            if name in self.mask_weight_indicator:
                self.mask_layer_dict[name.replace('.', '_')] = WeightLayer(param.data.shape[0], self.device)
                self.mask_weight_name.append('mask_layer_dict.' + name.replace('.', '_') + '.w')

    def weight_recover(self):
        for name, param in self.named_parameters():
            if name in self.mask_weight_indicator:
                param.data = param.data + self.bias_layer_dict[name]

    def stat_param_sizes(self):
        total_param_size = 0
        for name, param in self.named_parameters():
            if name in self.mask_weight_name or name.endswith('_mask'):
                continue
            elif name in self.mask_weight_indicator:
                total_param_size += (param.data != 0).float().sum()
            else:
                total_param_size += torch.numel(param.data)
        total_param_size = total_param_size * 32
        return total_param_size

    # for FedDST
    def _decay(self, t, alpha=0.3, t_end=400):
        if t >= t_end:
            return 0
        return alpha / 2 * (1 + np.cos(t * np.pi / t_end))

    def _weights_by_layer(self, sparsity=0.1, sparsity_distribution='uniform'):
        with torch.no_grad():
            layer_names = []
            for name in self.state_dict():
                if needs_mask(name, self.mask_weight_indicator) and not name.endswith('mask'):
                    layer_names.append(name)
            sparsities = np.empty(len(layer_names))
            n_weights = np.zeros_like(sparsities, dtype=int)

            i = 0
            for name, param in self.named_parameters():
                if needs_mask(name, self.mask_weight_indicator) and not name.endswith('mask'):
                    n_weights[i] += param.numel()
                    sparsities[i] = sparsity
                    i += 1
                    continue

            sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
            n_weights = np.floor((1 - sparsities) * n_weights)

            return {layer_names[i]: n_weights[i] for i in range(len(layer_names))}

    def layer_prune(self, sparsity=0.1, sparsity_distribution='erk'):
        '''
        Prune the network to the desired sparsity, following the specified
        sparsity distribution. The weight magnitude is used for pruning.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        '''

        # print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, param in self.named_parameters():

                # We need to figure out how many to prune
                n_total = 0
                if needs_mask(name, self.mask_weight_indicator) and not name.endswith('mask'):
                    n_total += param.numel()
                    n_prune = int(n_total - weights_by_layer[name])
                    if n_prune >= n_total or n_prune < 0:
                        continue
                        # Determine smallest indices
                    if not torch.where(param == 1.0, True, False).all():
                        _, prune_indices = torch.topk(torch.abs(param.data.flatten()),
                                                      n_prune, largest=False)
                    else:
                        prune_indices = torch.tensor(list(range(param.shape[0])[-n_prune:]))

                    # Write and apply mask
                    param.data.view(param.data.numel())[prune_indices] = 0
                    for bname, bparam in self.named_buffers():
                        if bname == name + '_mask':
                            bparam.view(bparam.numel())[prune_indices] = 0
                            break

    def layer_grow(self, sparsity=0.1, sparsity_distribution='erk'):
        '''
        Grow the network to the desired sparsity, following the specified
        sparsity distribution.
        The gradient magnitude is used for growing weights.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        '''

        # print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, param in self.named_parameters():
                if not needs_mask(name, self.mask_weight_indicator):
                    continue
                # We need to figure out how many to grow
                n_nonzero = 0
                for bname, buf in self.named_buffers():
                    if bname == name + '_mask':
                        n_nonzero += buf.count_nonzero().item()
                        break
                n_grow = int(weights_by_layer[name] - n_nonzero)
                if n_grow < 0:
                    continue
                # print('grow from', n_nonzero, 'to', weights_by_layer[name])

                _, grow_indices = torch.topk(torch.abs(param.grad.flatten()),
                                             n_grow, largest=True)

                # Write and apply mask
                param.data.view(param.data.numel())[grow_indices] = 0
                for bname, buf in self.named_buffers():
                    if bname == name + '_mask':
                        buf.view(buf.numel())[grow_indices] = 1
                        break
            # print('grown sparsity', self.sparsity())

    def sparsity(self, buffers=None):

        if buffers is None:
            buffers = self.named_buffers()

        n_ones = 0
        mask_size = 0
        for name, buf in buffers:
            if name.endswith('mask'):
                n_ones += torch.sum(buf)
                mask_size += buf.nelement()

        return 1 - (n_ones / mask_size).item()  # 返回的是参数稀疏度（也就是丢弃了多少参数的比例）

    def reset_weights(self, global_state=None, use_global_mask=False,
                      keep_local_masked_weights=False,
                      global_communication_mask=False):
        '''Reset weights to the given global state and apply the mask.
        - If global_state is None, then only apply the mask in the current state.
        - use_global_mask will reset the local mask to the global mask.
        - keep_local_masked_weights will use the global weights where masked 1, and
          use the local weights otherwise.
        '''

        with torch.no_grad():
            mask_changed = False
            local_state = self.state_dict()

            # If no global parameters were specified, that just means we should
            # apply the local mask, so the local state should be used as the
            # parameter source.
            if global_state is None:
                param_source = local_state
            else:
                param_source = global_state

            # We may wish to apply the global parameters but use the local mask.
            # In these cases, we will use the local state as the mask source.
            if use_global_mask:
                apply_mask_source = global_state
            else:
                apply_mask_source = local_state

            # We may wish to apply the global mask to the global parameters,
            # but not overwrite the local mask with it.
            if global_communication_mask:
                copy_mask_source = local_state
            else:
                copy_mask_source = apply_mask_source

            self.communication_sparsity = self.sparsity(apply_mask_source.items())

            # Empty new state to start with.
            new_state = {}

            # Copy over the params, masking them off if needed.
            for name, param in param_source.items():
                if name.endswith('_mask'):
                    # skip masks, since we will copy them with their corresponding
                    # layers, from the mask source.
                    continue

                new_state[name] = local_state[name]

                mask_name = name + '_mask'
                if needs_mask(name, self.mask_weight_indicator) and mask_name in apply_mask_source:

                    mask_to_apply = apply_mask_source[mask_name].to(device=self.device, dtype=torch.bool)
                    mask_to_copy = copy_mask_source[mask_name].to(device=self.device, dtype=torch.bool)
                    gpu_param = param[mask_to_apply].to(self.device)

                    # copy weights provided by the weight source, where the mask
                    # permits them to be copied
                    new_state[name][mask_to_apply] = gpu_param

                    # Don't bother allocating a *new* mask if not needed
                    if mask_name in local_state.keys():
                        new_state[mask_name] = local_state[mask_name]

                    new_state[mask_name].copy_(mask_to_copy)  # copy mask from mask_source into this model's mask

                    # what do we do with shadowed weights?
                    if not keep_local_masked_weights:
                        new_state[name][~mask_to_apply] = 0

                    if mask_name not in local_state or not torch.equal(local_state[mask_name], mask_to_copy):
                        mask_changed = True
                else:
                    # biases and other unmasked things
                    gpu_param = param.to(self.device)
                    new_state[name].copy_(gpu_param)

                # clean up copies made to gpu
                if gpu_param.data_ptr() != param.data_ptr():
                    del gpu_param

            self.load_state_dict(new_state)
        return mask_changed

    def clear_gradients(self):
        for _, layer in self.named_children():
            for _, param in layer.named_parameters():
                del param.grad
        torch.cuda.empty_cache()

    def prunefl_readjust(self, aggregate_gradients, layer_times, prunable_params=0.4):
        with torch.no_grad():
            importances = []
            masks = []
            n_grown = 0
            for i, g in enumerate(aggregate_gradients):
                g.square_()  # 梯度的平方作为重要性
                g = g.div(layer_times[i])
                importances.append(g)
                threshold = np.percentile(abs(g.cpu().numpy()), prunable_params * 100)
                mask = torch.where(torch.abs(g) < threshold, 0, 1)
                masks.append(mask)
                n_grown += mask.sum()
            cat_imp = torch.cat([torch.flatten(g) for g in importances])

            print('readj density', n_grown / cat_imp.numel())

            # set masks
            state = self.state_dict()
            i = 0
            n_differences = 0
            for name, param in state.items():
                if name.endswith('_mask'):
                    continue
                if not needs_mask(name, self.mask_weight_indicator):
                    continue

                n_differences += torch.count_nonzero(state[name + '_mask'].to('cpu') ^ masks[i].to('cpu'))
                state[name + '_mask'] = masks[i]
                i += 1

            print('mask changed percent', n_differences / cat_imp.numel())

            self.load_state_dict(state)
            return n_differences / cat_imp.numel()


class RNNModel(CommenNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, *args, **kwargs):
        super(RNNModel, self).__init__(*args, **kwargs)
        # related model parameters for RNN
        self.rnn_type = self.args.rnn_type
        self.d_embed = self.args.d_embed
        self.rnn_hidden = max(int(self.args.rnn_hidden * self.mask_rate), 1)
        self.rnn_layers = self.args.rnn_layers
        self.tie_weights = self.args.tie_weights

        # model structure
        self.encoder = nn.Embedding(self.args.nvocab, self.d_embed)
        assert self.rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if self.rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(self.d_embed if l == 0 else self.rnn_hidden,
                                       self.rnn_hidden if l != self.rnn_layers - 1 else (
                                           self.d_embed if self.tie_weights else self.rnn_hidden), 1)
                         for l in range(self.rnn_layers)]
        if self.rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(self.d_embed if l == 0 else self.rnn_hidden,
                                      self.rnn_hidden if l != self.rnn_layers - 1 else self.d_embed, 1)
                         for l in range(self.rnn_layers)]

        self.rnns = torch.nn.ModuleList(self.rnns)

        # initial model parameters
        if self.tie_weights:
            self.decoder = nn.Linear(self.d_embed, self.args.nvocab)
            self.init_weights()
            self.decoder.weight = self.encoder.weight
        else:
            self.decoder = nn.Linear(self.rnn_hidden, self.args.nvocab)
            self.init_weights()

        self.init_param_sizes()

    def init_weights(self):
        self.encoder.weight.data.normal_(mean=0, std=1)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(mean=0, std=1)

    def forward(self, input, hidden):
        x = self.encoder(input)
        new_hidden = []

        for l, rnn in enumerate(self.rnns):
            x, hidden_ = rnn(x, hidden[l])
            new_hidden.append(hidden_)
        hidden = new_hidden
        x = self.decoder(x)
        x = x.view((x.size(0) * x.size(1), x.size(2)))
        result = F.log_softmax(x, dim=1)
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.rnn_hidden if l != self.rnn_layers - 1 else (
                self.d_embed if self.tie_weights else self.rnn_hidden)).zero_().to(self.device),
                     weight.new(1, bsz, self.rnn_hidden if l != self.rnn_layers - 1 else (
                         self.d_embed if self.tie_weights else self.rnn_hidden)).zero_().to(self.device))
                    for l in range(self.rnn_layers)]

        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.rnn_hidden if l != self.rnn_layers - 1 else (
                self.d_embed if self.tie_weights else self.self.rnn_hidden)).zero_().to(self.device)
                    for l in range(self.rnn_layers)]


class _RNN(CommenNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, *args, **kwargs):
        super(_RNN, self).__init__(*args, **kwargs)
        # related model parameters for RNN
        self.rnn_type = self.args.rnn_type
        self.d_embed = self.args.d_embed
        self.rnn_hidden = max(int(self.args.rnn_hidden * self.mask_rate), 1)
        self.rnn_layers = self.args.rnn_layers
        self.tie_weights = self.args.tie_weights

        # model structure
        self.encoder = nn.Embedding(self.args.nvocab, self.d_embed)
        self.rnns = torch.nn.LSTM(self.d_embed, self.rnn_hidden, 2)

        self.decoder = nn.Linear(self.rnn_hidden, self.args.nvocab)

        # initial model parameters
        self.init_weights()
        self.init_param_sizes()

    def init_weights(self):
        self.encoder.weight.data.normal_(mean=0, std=1)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(mean=0, std=1)

    def forward(self, input):
        x = self.encoder(input)
        x, hidden_ = self.rnns(x)
        x = self.decoder(x)
        x = x.view((x.size(0) * x.size(1), x.size(2)))
        result = F.log_softmax(x, dim=1)
        return result


class MNISTNet(CommenNet):

    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, max(int(10 * self.mask_rate), 1), 5)  # "Conv 1-10"
        self.conv2 = nn.Conv2d(max(int(10 * self.mask_rate), 1), max(int(20 * self.mask_rate), 1), 5)  # "Conv 10-20"
        self.fc1 = nn.Linear(max(int(20 * self.mask_rate), 1) * 16, 50)
        self.fc2 = nn.Linear(50, 10)

        self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(CommenNet):
    def __init__(self, vgg_name, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.features = self._make_layers(cfg[vgg_name])
        if self.args.dataset == 'cifar100':
            self.classifier = nn.Linear(512, 100)
        elif self.args.dataset == 'tinyimagenet':
            self.classifier = nn.Linear(2048, 200)
        elif self.args.dataset == 'cifar10':
            self.classifier = nn.Linear(512, 10)

        self.init_param_sizes()
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}

    def set_adapted_para(self, name, val):
        self.adapted_model_para[name] = val

    def del_adapted_para(self):
        for key, val in self.adapted_model_para.items():
            if self.adapted_model_para[key] is not None:
                self.adapted_model_para[key].grad = None
                self.adapted_model_para[key] = None

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out)
        # return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        step = 0
        for x in cfg:
            step += 1
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if step == 1 or step == len(cfg) - 1:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x, momentum=None, track_running_stats=False),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, max(int(x * self.mask_rate), 1), kernel_size=3, padding=1),
                               nn.BatchNorm2d(max(int(x * self.mask_rate), 1), momentum=None,
                                              track_running_stats=False),
                               nn.ReLU(inplace=True)]
                if step < len(cfg) - 1 and step > 1:
                    in_channels = max(int(x * self.mask_rate), 1)
                else:
                    in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(args, device, mask_rate=1):
    return VGG('VGG11', args, device, mask_rate=mask_rate)


def VGG13(args, device, mask_rate=1):
    return VGG('VGG13', args, device, mask_rate=mask_rate)


def VGG16(args, device, mask_rate=1):
    return VGG('VGG16', args, device, mask_rate=mask_rate)


def VGG19(args, device, mask_rate=1):
    return VGG('VGG19', args, device, mask_rate=mask_rate)


all_models = {
    'mnist': MNISTNet,
    'cifar10': VGG11,
    'cifar100': VGG13,
    'tinyimagenet': VGG16,
    'reddit': RNNModel,
}
