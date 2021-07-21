#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="contrastive_mutate"

#python train.py \
#--autoencoder_type contrastive \
#--decoder_type selfie \
#--input_graph_transform_type mutate \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
