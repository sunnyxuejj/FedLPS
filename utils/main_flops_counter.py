import numpy as np

import torch
import torchvision
from torch.autograd import Variable
from util import repackage_hidden
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def count_training_flops(model, args, full=None):
    if full is None:
        full = not args.mask
    flops = 3 * count_model_param_flops(model, args, full=full)
    return flops

def count_inference_flops (model, args):
    flops = count_model_param_flops(model, args)
    return flops

def count_model_param_flops(model=None, args=None, multiply_adds=True, full=False):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        if not full:
            num_weight_params = (self.weight.data != 0).float().sum()
        else:
            num_weight_params = torch.numel(self.weight.data)
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
        if args.learnable:
            flops += self.weight.data.size(0)  # mask占用的flops
        # logging.info("-------")
        # logging.info("sparsity{}".format(num_weight_params/torch.numel(self.weight.data)))
        # logging.info("A{}".format(flops))
        list_conv.append(flops)

    list_lstm = []
    def lstm_hook(self, input, output):
        input_size, batch_size, hidden_size = input[0].shape[0], input[0].shape[1], input[0].shape[2]
        if not full:
            weight_hh_ops = (self.weight_hh_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
            weight_ih_ops = (self.weight_ih_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_hh_ops = (self.bias_hh_l0.data != 0).float().sum() if self.bias_hh_l0 is not None else 0
            bias_ih_ops = (self.bias_ih_l0.data != 0).float().sum() if self.bias_ih_l0 is not None else 0
        else:
            weight_hh_ops = torch.numel(self.weight_hh_l0.data) * (2 if multiply_adds else 1)
            weight_ih_ops = torch.numel(self.weight_ih_l0.data) * (2 if multiply_adds else 1)
            bias_hh_ops = torch.numel(self.bias_hh_l0.data) if self.bias_hh_l0 is not None else 0
            bias_ih_ops = torch.numel(self.bias_ih_l0.data) if self.bias_hh_l0 is not None else 0
        flops = batch_size * (weight_ih_ops + weight_hh_ops + bias_ih_ops + bias_hh_ops)
        if args.learnable:
            flops = flops + self.weight_hh_l0.data.size(0) + self.weight_ih_l0.data.size(0)  # mask占用的flops
        list_lstm.append(flops)

    list_embedding = []
    def embedding_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
        flops = batch_size * weight_ops
        if args.learnable:
            flops = flops + self.weight.data.size(0) # mask占用的flops
        list_embedding.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_ops = (self.bias.data != 0).float().sum() if self.bias is not None else 0
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
            bias_ops = torch.numel(self.bias.data) if self.bias is not None else 0
        flops = batch_size * (weight_ops + bias_ops)
        if args.learnable:
            flops = flops + self.weight.data.size(0) # mask占用的flops
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(handles, net):

        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handles += [net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                handles += [net.register_forward_hook(linear_hook)]
            if isinstance(net, torch.nn.LSTM):
                handles += [net.register_forward_hook(lstm_hook)]
            if isinstance(net, torch.nn.Embedding):
                handles += [net.register_forward_hook(embedding_hook)]
            return
        for c in childrens:
            foo(handles, c)

    # if model == None:
    #     model = torchvision.models.alexnet()
    handles = []
    foo(handles, model)
    model.eval()
    if args.dataset == "mnist" or args.dataset == "emnist":
        input_channel = 1
        input_res = 28
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(args.device)
        out = model(input)
    elif args.dataset == "cifar10":
        input_channel = 3
        input_res = 32
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(args.device)
        out = model(input)
    elif args.dataset == "cifar100":
        input_channel = 3
        input_res = 32
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(args.device)
        out = model(input)
    elif args.dataset == "tiny":
        input_channel = 3
        input_res = 64
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res).unsqueeze(0), requires_grad=True).to(args.device)
        out = model(input)
    elif args.dataset == "reddit":
        input_channel = 9
        input = Variable(torch.randint(high=model.args.nvocab, size=(input_channel, args.local_bs))).to(args.device)
        hidden = model.init_hidden(args.local_bs)
        hidden = repackage_hidden(hidden)
        out = model(input, hidden)
    elif args.dataset == "senti140":
        input_channel = args.nwords
        input = Variable(torch.randint(high=model.args.nvocab, size=(input_channel, args.local_bs))).to(args.device)
        hidden = model.init_hidden(args.local_bs)
        hidden = repackage_hidden(hidden)
        out = model(input, hidden)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample) + sum(list_lstm) + sum(list_embedding))
    for handle in handles:
        handle.remove()
    # print('  + Number of FLOPs: %.2f' % (total_flops))
    return total_flops

def count_flops_list(model=None, args=None, multiply_adds=True, full=False):

    prods = {}
    list_total = []

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        if not full:
            num_weight_params = (self.weight.data != 0).float().sum()
        else:
            num_weight_params = torch.numel(self.weight.data)
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
        # if not full:
        #     flops += self.weight.data.size(0)  # mask占用的flops
        list_conv.append(flops)
        list_total.append(flops)

    list_lstm = []
    def lstm_hook(self, input, output):
        input_size, batch_size, hidden_size = input[0].shape[0], input[0].shape[1], input[0].shape[2]
        if not full:
            weight_hh_ops = (self.weight_hh_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
            weight_ih_ops = (self.weight_ih_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_hh_ops = (self.bias_hh_l0.data != 0).float().sum() if self.bias_hh_l0 is not None else 0
            bias_ih_ops = (self.bias_ih_l0.data != 0).float().sum() if self.bias_ih_l0 is not None else 0
        else:
            weight_hh_ops = torch.numel(self.weight_hh_l0.data) * (2 if multiply_adds else 1)
            weight_ih_ops = torch.numel(self.weight_ih_l0.data) * (2 if multiply_adds else 1)
            bias_hh_ops = torch.numel(self.bias_hh_l0.data) if self.bias_hh_l0 is not None else 0
            bias_ih_ops = torch.numel(self.bias_ih_l0.data) if self.bias_hh_l0 is not None else 0
        flops = batch_size * (weight_ih_ops + weight_hh_ops + bias_ih_ops + bias_hh_ops)
        # if not full:
        #     flops = flops + self.weight_hh_l0.data.size(0) + self.weight_ih_l0.data(0)  # mask占用的flops
        list_lstm.append(flops)
        list_total.append(flops)

    list_embedding = []
    def embedding_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
        flops = batch_size * weight_ops
        # logging.info("L{}".format(flops))
        list_embedding.append(flops)
        list_total.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_ops = (self.bias.data != 0).float().sum() if self.bias is not None else 0
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
            bias_ops = torch.numel(self.bias.data) if self.bias is not None else 0
        flops = batch_size * (weight_ops + bias_ops)
        # if not full:
        #     flops = flops + self.weight.data.size(0) # mask占用的flops
        list_linear.append(flops)
        list_total.append(flops)


    def foo(handles, net):

        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handles += [net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                handles += [net.register_forward_hook(linear_hook)]
            if isinstance(net, torch.nn.LSTM):
                handles += [net.register_forward_hook(lstm_hook)]
            if isinstance(net, torch.nn.Embedding):
                handles += [net.register_forward_hook(embedding_hook)]
            return
        for c in childrens:
            foo(handles, c)

    # if model == None:
    #     model = torchvision.models.alexnet()
    handles = []
    foo(handles, model)
    model.eval()
    device = next(model.parameters()).device
    if args.dataset == "mnist" or args.dataset == "emnist":
        input_channel = 1
        input_res = 28
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(device)
        out = model(input)
    elif args.dataset == "cifar10":
        input_channel = 3
        input_res = 32
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(device)
        out = model(input)
    elif args.dataset == "cifar100":
        input_channel = 3
        input_res = 32
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res), requires_grad=True).to(device)
        out = model(input)
    elif args.dataset == "tiny":
        input_channel = 3
        input_res = 64
        input = Variable(torch.rand(args.local_bs, input_channel, input_res, input_res).unsqueeze(0), requires_grad=True).to(device)
        out = model(input)
    elif args.dataset == "reddit":
        input_channel = 9
        input = Variable(torch.randint(high=model.args.nvocab, size=(input_channel, args.local_bs))).to(device)
        hidden = model.init_hidden(args.local_bs)
        hidden = repackage_hidden(hidden)
        out = model(input, hidden)

    for handle in handles:
        handle.remove()

    return list_total
