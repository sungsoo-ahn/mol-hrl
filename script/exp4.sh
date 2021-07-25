#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="variational"

python train.py \
--autoencoder_type variational \
--code_dim 256
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
