#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="smiles"

python train.py \
--autoencoder_type base \
--decoder_type smiles \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
