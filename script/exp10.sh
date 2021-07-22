#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2smiles_supervised"

python train.py \
--autoencoder_type supervised \
--encoder_type graph \
--decoder_type smiles \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
