#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="relational_mutate"

python train.py \
--autoencoder_type relational \
--decoder_type selfie \
--input_graph_transform_type mutate \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
