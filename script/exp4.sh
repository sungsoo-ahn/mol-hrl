#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

#python run_condopt.py \
#--train_split train_01 \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condopt_01"

#python run_condopt.py \
#--train_split train_05 \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condopt_05"

python run_conddecoder.py \
--train_split train_001 \
--cond_embedding_mlp \
--score_func_name logp \
--tag "${TAG}_condgen_001_logp"

#python run_conddecoder.py \
#--train_split train_01 \
#--score_func_name molwt \
#--tag "${TAG}_condgen_01_molwt"
#
#python run_conddecoder.py \
#--train_split train_01 \
#--score_func_name qed \
#--tag "${TAG}_condgen_01_qed"
#
#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name logp \
#--tag "${TAG}_condgen_05_logp"
#
#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name molwt \
#--tag "${TAG}_condgen_05_molwt"
#
#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name qed \
#--tag "${TAG}_condgen_05_qed"