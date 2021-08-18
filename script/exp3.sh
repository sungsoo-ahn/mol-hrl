#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 128 \
--dataset_name maskgraph2seq \
--tags vq lstm maskgraph2seq \
--checkpoint_path ../resource/checkpoint/vq_lstm_maskgraph2seq