#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="relational_abl"

#python train.py \
#--autoencoder_type relational_abl \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
