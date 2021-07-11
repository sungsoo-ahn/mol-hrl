#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_vmf0"

#python train_ae.py \
#--ae_type sae \
#--sae_vmf_scale 1.0 \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
