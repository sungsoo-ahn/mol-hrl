#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="codedim3"

python train.py \
--autoencoder_type base \
--code_dim 256
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
