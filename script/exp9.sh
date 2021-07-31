#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python train.py \
--autoencoder_type base \
--input_graph_fragment \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag "${TAG}_linear"
python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG