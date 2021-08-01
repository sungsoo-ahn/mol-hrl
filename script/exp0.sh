#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--score_func_name penalized_logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--score_func_name molwt \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--score_func_name qed \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG