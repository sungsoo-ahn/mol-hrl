#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="relational"

#python train.py \
#--autoencoder_type relational \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
