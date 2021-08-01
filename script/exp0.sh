#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python run_conddecoder.py \
--score_func_name penalized_logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_penalized_logp.pth" \
--tag "${TAG}_penalized_logp"

python run_conddecoder.py \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_logp.pth" \
--tag "${TAG}_logp"

python run_conddecoder.py \
--score_func_name molwt \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_molwt.pth" \
--tag "${TAG}_molwt"

python run_conddecoder.py \
--score_func_name qed \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_qed.pth" \
--tag "${TAG}_qed"
