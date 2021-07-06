#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_ae_mutate"

python train_ae.py \
--ae_type ae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_mutate \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG seq