#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_ae_mask3"

python train_ae.py \
--ae_type ae \
--mask_rate 0.3 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG