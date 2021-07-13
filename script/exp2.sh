#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="rae"

python train_ae.py \
--ae_type rae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
