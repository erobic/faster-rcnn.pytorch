#!/usr/bin/env bash

source activate vqa

CUDA_VISIBLE_DEVICES=0 python -u test_net.py \
--dataset CLEVR \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--num_images 1000

# Takes 1115 MB GPU mem