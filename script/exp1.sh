#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_plug.py \
--train_split train \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_plug_full.pth" \
--tags $TAG full