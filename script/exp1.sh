#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="supervised"

#python train.py \
#--autoencoder_type supervised \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag "${TAG}_linear"
python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG