#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

#python run_condopt.py \
#--train_split train_001 \
#--cond_embedding_mlp \
#--tag "${TAG}_condopt_001"

python run_conddecoder.py \
--train_split train_256 \
--score_func_name logp \
--cond_embedding_mlp \
--tags $TAG 256 logp