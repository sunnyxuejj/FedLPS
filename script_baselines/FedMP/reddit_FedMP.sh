#!/bin/bash
cd ../../

python FedMP.py --dataset 'reddit' \
--nusers 100 \
--frac 0.1 \
--local_ep 1 \
--local_bs 5 \
--momentum 0.5 \
--clip 1 \
--lr 8
