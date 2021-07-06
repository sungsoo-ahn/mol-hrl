#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_ae_mask"

python train_ae.py \
--ae_type ae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--mask_rate 0.2 \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG seq