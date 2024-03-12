#!/bin/bash
cd ../../

python Hermes.py --dataset 'reddit' \
--nusers 100 \
--frac 0.1 \
--local_ep 1 \
--local_bs 5 \
--momentum 0.5 \
--clip 1 \
--online_decision True \
--lr 8 \
--prune_start_acc 0.15 \
--prune_end_rate 0.5
