#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="codedim0"

python train.py \
--autoencoder_type base \
--code_dim 32
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG