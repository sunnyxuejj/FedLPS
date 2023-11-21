# -*- coding: utf-8 -*-
# @python: 3.8

import copy
import torch
import numpy as np


def avg(w, w_server, args, device):
    """
    aggregate based on gradient
    """

    grad = copy.deepcopy(w)
    for i in range(len(w)):
        for k in w[0]:
            w[i][k] = grad[i][k] + w_server[k]

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)) + torch.mul(torch.randn(w_avg[k].shape).to(device), args.dp)
    return w_avg

def average_w(w):
    """
    aggregate based on weights
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights(w, avg_weight, args):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w[0])
    avg_weight = np.array(avg_weight)
    avg_weight = torch.Tensor(list(avg_weight / sum(avg_weight))).to(args.device)
    for i in range(len(w)):
        for k in w[i]:
            w[i][k] = w[i][k] * avg_weight[i]
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w[0][k]).to(args.device)
        for i in range(len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = w_avg[k] + torch.mul(torch.randn(w_avg[k].shape), args.dp).to(args.device)
    return w_avg
