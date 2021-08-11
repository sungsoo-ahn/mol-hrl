#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_plug.py \
--train_split train \
--score_func_name logp \
--plug_beta 0.1 \
--plug_code_dim 256 \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp largebeta largedim