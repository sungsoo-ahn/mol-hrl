#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

python run_conddecoder.py \
--train_split train \
--score_func_name logp \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG full logp

python run_conddecoder.py \
--train_split train \
--score_func_name molwt \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG full molwt

python run_conddecoder.py \
--train_split train \
--score_func_name qed \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG full qed
