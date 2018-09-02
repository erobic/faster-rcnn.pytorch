#!/usr/bin/env bash

source activate vqa

ROOT=/hdd/robik
DATASET=CLEVR

FASTER_RCNN_DIR=$ROOT/FasterRCNN

CUDA_VISIBLE_DEVICES=1 python -u demo.py \
--dataset $DATASET \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--load_dir $FASTER_RCNN_DIR/models \
--image_dir $FASTER_RCNN_DIR/demo_images \
--out_dir $FASTER_RCNN_DIR/out_demo_images