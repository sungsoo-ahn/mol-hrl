#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python train.py \
--autoencoder_type base \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG