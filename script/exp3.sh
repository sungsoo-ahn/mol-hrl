#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="srae"

python train_ae.py \
--ae_type srae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
