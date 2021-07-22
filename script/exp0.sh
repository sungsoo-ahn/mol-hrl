#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2smiles"

#python train.py \
#--autoencoder_type base \
#--encoder_type graph \
#--decoder_type smiles \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
