#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="base"

#python run_condopt.py \
#--train_split train_256 \
#--freeze_decoder \
#--cond_embedding_mlp \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condopt_001"

python run_conddecoder.py \
--train_split train_256 \
--score_func_name logp \
--freeze_decoder \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp

#python run_conddecoder.py \
#--train_split train_01 \
#--score_func_name molwt \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condgen_01_molwt"

#python run_conddecoder.py \
#--train_split train_01 \
#--score_func_name qed \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condgen_01_qed"

#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name logp \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condgen_05_logp"

#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name molwt \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condgen_05_molwt"

#python run_conddecoder.py \
#--train_split train_05 \
#--score_func_name qed \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag "${TAG}_condgen_05_qed"