#!/usr/bin/env bash
set -e
source activate vqa
python -u generate_clevr_bb.py --dataset CLEVR-CoGenT-A