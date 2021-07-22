#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2smiles_relational_mutate"

#python train.py \
#--autoencoder_type relational \
#--encoder_type graph \
#--decoder_type smiles \
#--input_graph_transform_type mutate \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
