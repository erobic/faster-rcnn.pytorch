#!/usr/bin/env bash
set -e

source activate vqa2
ROOT=/hdd/robik
DATASET=CLEVR-CoGenT-B
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
--load_subdir clevr