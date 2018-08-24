#!/usr/bin/env bash
source activate vqa
ROOT=/hdd/robik
DATASET=CLEVR
DATA_ROOT=${ROOT}/${DATASET}

SPLIT=train
#python -u extract_features.py --dataset $DATASET \
#--image_dir $DATA_ROOT/images/${SPLIT}/

python -u extract_features.py --dataset $DATASET \
--root $ROOT \
--split $SPLIT \
--image_dir $DATA_ROOT/demo_images/ \
--net res101 \
--checksession 1 \
--checkepoch 3 \
--checkpoint 34999 \
--cuda \
