#!/bin/bash
cd ../

python FedLPS.py --dataset 'cifar100' \
--nclass 10 \
--nsamples 20 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--lr 0.1
