#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--input_fragment \
--tag $TAG

python run_conddecoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG