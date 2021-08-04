#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="fragment2"

#python run_autoencoder.py \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--input_fragment2 \
#--tag $TAG

python run_conddecoder.py \
--train_split train_256 \
--score_func_name logp \
--freeze_decoder \
--cond_embedding_mlp \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tags $TAG 256 logp