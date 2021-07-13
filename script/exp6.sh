#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_seq_uni0"

python train_ae.py \
--ae_type sae \
--encoder_type seq \
--sae_uniform_loss_coef 0.01 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
