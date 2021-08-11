#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_plug.py \
--train_split train \
--score_func_name logp \
--plug_depth 3 \
--plug_width_factor 4.0 \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp