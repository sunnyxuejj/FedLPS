#!/bin/bash
cd ../../

python Hermes.py --dataset 'mnist' \
--nclass 2 \
--nsamples 100 \
--nusers 100 \
--frac 0.1 \
--local_ep 5 \
--local_bs 32 \
--lr 0.1 \
--prune_start_acc 0.5 \
--prune_end_rate 0.5
