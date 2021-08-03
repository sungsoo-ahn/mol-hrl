#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="hyper1"

python run_conddecoder.py \
--train_split train_01 \
--lr 1e-3 \
--cond_embedding_mlp \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/base.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_logp"