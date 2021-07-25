#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="masking"

python train.py \
--autoencoder_type masking \
--input_graph_subgraph \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
