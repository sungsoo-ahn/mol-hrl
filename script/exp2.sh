#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="selfie"

python train.py \
--autoencoder_type base \
--decoder_type selfie \
--input_graph_transform_type mask \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
