#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="sae3"

python train_ae.py \
--ae_type sae \
--seq_decoder_num_layers 2 \
--seq_decoder_dropout 0.5 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
