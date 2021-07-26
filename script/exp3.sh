#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="contrastive"

python train.py \
--autoencoder_type contrastive \
--input_graph_mutate \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
