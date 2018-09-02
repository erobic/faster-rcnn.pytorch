#!/usr/bin/env bash
set -e

source activate vqa
ROOT=/hdd/robik
DATASET=CLEVR
DATA_ROOT=${ROOT}/${DATASET}

SPLIT=val
CUDA_VISIBLE_DEVICES=0 python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--visualize_only \
--image_limit 2 \
--visualize_subdir visualize_oracle \
--use_oracle_gt_boxes