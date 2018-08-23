#!/usr/bin/env bash

source activate vqa
GPU_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python -u trainval_net.py \
                   --dataset clevr --net res101 \
                   --bs 4 --nw 4 \
                   --lr 0.001 --lr_decay_step 5 \
                   --cuda