#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="variational"

#python train.py \
#--autoencoder_type variational \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag "${TAG}_linear"
python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG