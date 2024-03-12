#!/bin/bash
cd ../../

python Hermes.py --dataset 'tinyimagenet' \
--nclass 20 \
--nsamples 40 \
--nusers 50 \
--frac 0.2 \
--local_ep 5 \
--local_bs 20 \
--momentum 0 \
--lr 0.1 \
--prune_start_acc 0.06 \
--prune_end_rate 0.4
