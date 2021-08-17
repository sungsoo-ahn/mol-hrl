#!/bin/bash

python run_autoencoder.py \
--vq \
--vq_dim 256 \
--code_dim 256 \
--lr 1e-4 \
--tags vq \
--checkpoint_path ../resource/checkpoint/vq