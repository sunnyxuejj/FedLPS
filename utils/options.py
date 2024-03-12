#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8

import os
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--split', type=int, default=0, help='do test split')
    parser.add_argument('--num_classes', default=10, type=int, help='number of image classes in all dataset')
    parser.add_argument('--nclass', type=int, default=2, help="number of image classes per client have")
    parser.add_argument('--nsamples', type=int, default=100, help="number of images per class per client have")
    parser.add_argument('--rate_unbalance', type=float, default=1.0, help="unbalance rate")

    # learning arguments
    parser.add_argument('--rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--nusers', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use (sgd, adam)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate client')
    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')
    parser.add_argument('--clip', type=float, default=0, help='gradient clipping')
    parser.add_argument('--momentum', type=float, default=0, help='SGD momentum (default: 0.5)')
    parser.add_argument('--epsilon', type=float, default=1.2, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')

    # hyperparameters
    # for RNN
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on '
                             'RNN activiation (beta = 0 means no regularization)')
    # for our FedLPS
    parser.add_argument('--wd', type=float, default=1, help='loss corr wd')
    parser.add_argument('--lamda_mask', type=float, default=0.0001, help='loss corr lamda_mask')

    # model arguments
    parser.add_argument('--d_embed', type=int, default=200, help='embedding dimension')
    parser.add_argument('--d_dict', type=int, default=10000, help='size of the dictionary of embeddings')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    # for RNN
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='type of RNN')
    parser.add_argument("--rnn_hidden", type=int, default=256, help="RNN hidden unit dimensionality")
    parser.add_argument("--rnn_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument('--tie_weights', type=bool, default=True, help='')

    # method arguments
    parser.add_argument('--agg', type=str, default='avg', help='averaging strategy')
    parser.add_argument('--fedprox', type=bool, default=False, help='whether fedprox')
    parser.add_argument('--lamda', type=float, default=0.001, help='the regulization coffient of fedprox and Hermes')
    parser.add_argument('--sample_rate', type=float, default=0.5, help='')
    parser.add_argument('--mask', type=bool, default=False, help='used mask')
    parser.add_argument("--learnable", type=bool, default=False)
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask rate')

    # Pruning and regrowth options
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity from 0 to 1')
    parser.add_argument('--rate-decay-method', default='constant', choices=('constant', 'cosine'),
                        help='annealing for readjustment ratio')
    parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')
    parser.add_argument('--readjustment-ratio', type=float, default=0.1,
                        help='readjust this many of the weights each time')
    parser.add_argument('--pruning-begin', type=int, default=2, help='first epoch number when we should readjust')
    parser.add_argument('--pruning-interval', type=int, default=6, help='epochs between readjustments')
    parser.add_argument('--rounds-between-readjustments', type=int, default=5, help='rounds between readjustments')
    parser.add_argument('--remember-old', default=False, action='store_true',
                        help="remember client's old weights when aggregating missing ones")
    parser.add_argument('--sparsity-distribution', default='uniform', choices=('uniform', 'er', 'erk'))
    parser.add_argument('--final-sparsity', type=float, default=None,
                        help='final sparsity to grow to, from 0 to 1. default is the same as --sparsity')
    # Pruning options for prunefl
    parser.add_argument('--initial-rounds', default=10, type=int, help='number of "initial pruning" rounds for prunefl')
    # pruning options for LotteryFL
    parser.add_argument('--prune_percent', type=float, default=10,
                        help='pruning percent')
    parser.add_argument('--prune_start_acc', type=float, default=0.5,
                        help='pruning start acc')
    parser.add_argument('--prune_end_rate', type=float, default=0.5,
                        help='pruning end rate')

    # roll option for FedRolex
    parser.add_argument('--overlap', default=None, type=float)
    parser.add_argument('--roll_rate', default=0.5, type=float)
    parser.add_argument('--model_split_mode', default='dynamic', type=str)

    # Bayesian argument
    parser.add_argument("--weight_scale", type=float, default=0.01)
    parser.add_argument("--rho_offset", type=int, default=-2.5)
    parser.add_argument("--zeta", type=int, default=0.5)

    # dropout rate online decision
    parser.add_argument("--online_decision", type=bool, default=False)

    # other arguments
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--log_interval', type=int, default=3, metavar='N',
                        help='report interval')
    args = parser.parse_args()
    return args
