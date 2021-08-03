#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="hyper4"

python run_conddecoder.py \
--train_split train_01 \
--lr 1e-3 \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_logp"