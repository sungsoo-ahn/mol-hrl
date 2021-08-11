#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base_l2"

python run_autoencoder.py \
--l2_coef 1e-3 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags base l2