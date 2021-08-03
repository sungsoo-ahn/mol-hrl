#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="rgroup"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--input_rgroup \
--tag $TAG