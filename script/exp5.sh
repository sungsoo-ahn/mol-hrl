#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_graph_uni1"

python train_ae.py \
--ae_type sae \
--encoder_type graph \
--sae_uniform_loss_coef 0.001 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_graph_uni1_random"

python train_ae.py \
--ae_type sae \
--encoder_type graph \
--sae_uniform_loss_coef 0.001 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_random_smiles \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
