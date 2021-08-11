#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_plug.py \
--train_split train_001 \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_plug_001.pth" \
--tags $TAG 001