#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="smiles2smiles_style"

python train.py \
--autoencoder_type style \
--encoder_type smiles \
--decoder_type smiles \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
