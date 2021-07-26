#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="denoising1"

python train.py \
--autoencoder_type base \
--input_graph_fragment_contract \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
