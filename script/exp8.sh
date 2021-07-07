#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_sae"

python train_ae.py \
--ae_type sae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
