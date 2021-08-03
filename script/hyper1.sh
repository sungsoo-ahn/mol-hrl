#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="hyper1"

python run_condopt.py \
--train_split train_01 \
--num_warmup_steps 200 \
--load_checkpoint_path "${CHECKPOINT_DIR}/mutate.pth" \
--tag "${TAG}"