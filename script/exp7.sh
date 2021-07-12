#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_attack1_uni1"

python train_ae.py \
--ae_type sae \
--sae_attack_steps 3 \
--sae_uniform_loss_coef 0.1 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
