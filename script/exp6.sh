#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2seq_ae_mutate"

python train_ae.py \
--ae_type ae \
--encoder_type graph \
--use_mutate \
--dm_type graph2seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth"
