#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="nopretrain"

python run_conddecoder.py \
--train_split train_256 \
--score_func_name logp \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG 256 logp

python run_conddecoder.py \
--train_split train_256 \
--score_func_name molwt \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG 256 molwt

python run_conddecoder.py \
--train_split train_256 \
--score_func_name qed \
--cond_embedding_mlp \
--max_epochs 100 \
--check_val_every_n_epoch 10 \
--tags $TAG 256 qed
