#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_aae"

python train_ae.py \
--ae_type aae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG seq