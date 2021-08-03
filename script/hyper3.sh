#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="hyper3"

python run_condopt.py \
--train_split train_01 \
--num_warmup_steps 500 \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag "${TAG}"