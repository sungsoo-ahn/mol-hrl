#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 32 \
--dataset_name fraggraph2seq \
--tags vq lstm fraggraph2seq 32 \
--checkpoint_path ../resource/checkpoint/vq_lstm_fraggraph2seq_32