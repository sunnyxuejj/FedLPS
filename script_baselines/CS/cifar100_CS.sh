#!/bin/bash
cd ../../

python CS.py --dataset 'cifar100' \
--nclass 10 \
--nsamples 20 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--momentum 0.5 \
--lr 0.1 \
--mask_rate 0.5
