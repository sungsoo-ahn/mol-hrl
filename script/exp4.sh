#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_cae"

python train_ae.py \
--ae_type cae \
--dm_type "seqs2seq" \
--mask_rate 0.1 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG