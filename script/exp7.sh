#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="smiles2smiles_denoising_mask"

python train.py \
--autoencoder_type base \
--encoder_type smiles \
--decoder_type smiles \
--input_sequence_transform_type mask \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
