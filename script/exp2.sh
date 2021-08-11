#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment"

python run_conddecoder.py \
--train_split train_256 \
--freeze_decoder \
--score_func_name logp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp