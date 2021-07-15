#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="ae_seq"

python train_ae.py \
--ae_type ae \
--encoder_type seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG