#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

#python run_conddecoder.py \
#--train_split train_256 \
#--score_func_name logp \
#--cond_embedding_mlp \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tags $TAG 256 logp

python run_conddecoder.py \
--train_split train_256 \
--score_func_name molwt \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 molwt

python run_conddecoder.py \
--train_split train_256 \
--score_func_name qed \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 qed
