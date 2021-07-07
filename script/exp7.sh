#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2seq_cae_mutate"

python train_ae.py \
--ae_type cae \
--encoder_type graph \
--use_mutate \
--dm_type graphs2seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG

