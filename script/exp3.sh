#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

python run_conddecoder.py \
--decoder_lr 1e-3 \
--score_func_name penalized_logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--decoder_lr 1e-3 \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--decoder_lr 1e-3 \
--score_func_name molwt \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--decoder_lr 1e-3 \
--score_func_name qed \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--decoder_lr 1e-3 \
--train_split train \
--score_func_name penalized_logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG
