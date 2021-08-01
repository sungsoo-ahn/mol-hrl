#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG