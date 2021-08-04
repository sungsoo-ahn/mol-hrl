#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_autoencoder.py \
--code_dim 1024 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_large.pth" \
--tags $TAG "large"
