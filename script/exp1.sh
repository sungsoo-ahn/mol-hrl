#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="mutate"

#python run_condopt.py \
#--train_split train_256 \
#--freeze_decoder \
#--cond_embedding_mlp \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condopt_256"

python run_conddecoder.py \
--train_split train_256 \
--score_func_name logp \
--freeze_decoder \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp