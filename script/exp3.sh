#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="selfie2selfie"

python train.py \
--autoencoder_type base \
--encoder_type selfie \
--decoder_type selfie \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
