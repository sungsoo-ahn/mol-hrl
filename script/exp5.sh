#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae_seq"

python train_ae.py \
--ae_type sae \
--encoder_type seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
