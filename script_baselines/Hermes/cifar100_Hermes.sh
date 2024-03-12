#!/bin/bash
cd ../../

python Hermes.py --dataset 'cifar100' \
--nclass 10 \
--nsamples 20 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--momentum 0.5 \
--lr 0.1 \
--prune_start_acc 0.25 \
--prune_end_rate 0.3
