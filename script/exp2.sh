#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_uniform0"

#python train_ae.py \
#--ae_type sae \
#--sae_uniform_loss_coef 1e-2 \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
