#!/bin/bash
cd ../../

python FedMP.py --dataset 'tinyimagenet' \
--nclass 20 \
--nsamples 40 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--momentum 0 \
--lr 0.1
