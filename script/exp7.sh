#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_autoencoder.py \
--code_dim 1024 \
--input_fragment2 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_large.pth" \
--tags $TAG "large"
