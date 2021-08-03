#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="hyper3"

python run_conddecoder.py \
--train_split train_01 \
--lr 1e-4 \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/base.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_logp"