#!/bin/bash

python run_autoencoder.py \
--vq \
--vq_dim 256 \
--code_dim 256 \
--lr 1e-4 \
--decoder_name transformer_base \
--tags vq transformer \
--checkpoint_path ../resource/checkpoint/vq_transformer