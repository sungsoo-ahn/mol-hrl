#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_plug.py \
--train_split train_010 \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_plug_010.pth" \
--tags $TAG 010