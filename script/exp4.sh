#!/bin/bash

python run_autoencoder.py \
--code_dim 128 \
--dataset_name fraggraph2seq \
--tags lstm fraggraph2seq code128 \
--checkpoint_path ../resource/checkpoint/lstm_fraggraph2seq_code128
