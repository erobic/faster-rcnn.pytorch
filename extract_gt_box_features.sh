#!/usr/bin/env bash
set -e

source activate vqa
ROOT=/hdd/robik
DATASET=CLEVR
DATA_ROOT=${ROOT}/${DATASET}

SPLIT=train
CUDA_VISIBLE_DEVICES=2 python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--use_oracle_gt_boxes \
--bs 128

SPLIT=val
CUDA_VISIBLE_DEVICES=2 python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--use_oracle_gt_boxes \
--bs 128