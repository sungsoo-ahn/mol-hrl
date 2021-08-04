#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_condopt.py \
--train_split train_256 \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 penalized_logp
