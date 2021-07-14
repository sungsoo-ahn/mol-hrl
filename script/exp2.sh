#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="rae_graph"

python train_ae.py \
--ae_type rae \
--encoder_type graph \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG

CHECKPOINT_DIR="../resource/checkpoint"
TAG="rae_graph_random"

python train_ae.py \
--ae_type rae \
--encoder_type graph \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_random_smiles \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
