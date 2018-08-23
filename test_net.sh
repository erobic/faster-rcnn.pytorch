#!/usr/bin/env bash

source activate vqa

CUDA_VISIBLE_DEVICES=2 python test_net.py \
--dataset clevr \
--net res101 \
--checksession 1 \
--checkepoch 2 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models
# --num_images 10