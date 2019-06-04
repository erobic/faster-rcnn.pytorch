#!/usr/bin/env bash
set -e

ROOT=/hdd/robik
DATASET=CLEVR
DATA_ROOT=${ROOT}/${DATASET}
mkdir -p ${DATA_ROOT}/features

SPLIT=train
CUDA_VISIBLE_DEVICES=0 python -u extract_resnet_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--load_subdir clevr

SPLIT=val
CUDA_VISIBLE_DEVICES=0 python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--load_subdir clevr

SPLIT=test
CUDA_VISIBLE_DEVICES=0 python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--net res101 \
--checksession 1 \
--checkepoch 11 \
--checkpoint 34999 \
--cuda \
--load_dir /hdd/robik/FasterRCNN/models \
--load_subdir clevr