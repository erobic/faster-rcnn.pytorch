#!/usr/bin/env bash

source activate vqa

ROOT=/hdd/robik
DATASET=CLEVR


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python extract_features.py --net res101 \
--checksession 1 \
--checkepoch 2 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--dataset $DATASET \
--image_dir /hdd/robik/CLEVR/demo_images \
--root $ROOT \
--split test