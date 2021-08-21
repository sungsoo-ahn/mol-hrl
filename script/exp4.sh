#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 16 \
--dataset_name fraggraph2seq \
--tags vq lstm fraggraph2seq 16 \
--checkpoint_path ../resource/checkpoint/vq_lstm_fraggraph2seq_16