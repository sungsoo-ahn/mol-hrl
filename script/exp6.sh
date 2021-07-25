#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="contrastive1"

python train.py \
--autoencoder_type contrastive \
--input_graph_fragment_contract \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
