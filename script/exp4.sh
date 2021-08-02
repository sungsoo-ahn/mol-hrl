#! /bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

python run_conddecoder.py \
--train_split train_01 \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_plogp"

python run_condopt.py \
--train_split train_01 \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}_01_logp.pth" \
--tag "${TAG}_01_plogp"

python run_conddecoder.py \
--train_split train_10 \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_10_logp.pth" \
--tag "${TAG}_10_plogp"

python run_condopt.py \
--train_split train_10 \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}_10_logp.pth" \
--tag "${TAG}_10_plogp"

python run_conddecoder.py \
--train_split train \
--score_func_name logp \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_00_logp.pth" \
--tag "${TAG}_00_plogp"

python run_condopt.py \
--train_split train \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}_00_logp.pth" \
--tag "${TAG}_00_plogp"
