#!/bin/bash
cd ../../

python Hermes.py --dataset 'cifar10' \
--nclass 2 \
--nsamples 100 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--lr 0.1 \
--prune_start_acc 0.45 \
--prune_end_rate 0.3
