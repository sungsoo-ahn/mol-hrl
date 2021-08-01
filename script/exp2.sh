#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--input_fragment \
--tag $TAG

python run_conddecoder.py \
--score_func_name penalized_logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag "${TAG}_penalized_logp"

python run_conddecoder.py \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag "${TAG}_logp"

python run_conddecoder.py \
--score_func_name molwt \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag "${TAG}_molwt"

python run_conddecoder.py \
--score_func_name qed \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag "${TAG}_qed"
