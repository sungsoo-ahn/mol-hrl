#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="mutate"

python run_autoencoder.py \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--input_mutate \
--tag $TAG

python run_conddecoder.py \
--train_split train_01 \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_logp"

python run_conddecoder.py \
--train_split train_01 \
--score_func_name molwt \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_molwt.pth" \
--tag "${TAG}_01_molwt"

python run_conddecoder.py \
--train_split train_01 \
--score_func_name qed \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_qed.pth" \
--tag "${TAG}_01_qed"

python run_conddecoder.py \
--train_split train_10 \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_10_logp.pth" \
--tag "${TAG}_10_logp"

python run_conddecoder.py \
--train_split train_10 \
--score_func_name molwt \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_10_molwt.pth" \
--tag "${TAG}_10_molwt"

python run_conddecoder.py \
--train_split train_10 \
--score_func_name qed \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_10_qed.pth" \
--tag "${TAG}_10_qed"

python run_conddecoder.py \
--train_split train \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_00_logp.pth" \
--tag "${TAG}_00_logp"

python run_conddecoder.py \
--train_split train \
--score_func_name molwt \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_00_molwt.pth" \
--tag "${TAG}_00_molwt"

python run_conddecoder.py \
--train_split train \
--score_func_name qed \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_00_qed.pth" \
--tag "${TAG}_00_qed"