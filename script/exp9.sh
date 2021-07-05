#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2seq_cae"

python train_ae.py \
--ae_type cae \
--encoder_type graph \
--use_mutate \
--dm_type graphs2seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--max_epochs 1 \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG graph