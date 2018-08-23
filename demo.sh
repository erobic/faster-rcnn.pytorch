#!/usr/bin/env bash

source activate vqa

CUDA_VISIBLE_DEVICES=2 python demo.py --net res101 \
--checksession 1 \
--checkepoch 2 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--dataset clevr \
--image_dir /hdd/robik/CLEVR/demo_images