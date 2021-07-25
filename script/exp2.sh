#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="codedim2"

python train.py \
--autoencoder_type base \
--code_dim 128 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
