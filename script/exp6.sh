#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="denoising_mask"

#python train.py \
#--autoencoder_type base \
#--input_graph_mask \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
