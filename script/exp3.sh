#!/bin/bash

python run_autoencoder.py \
--code_dim 64 \
--dataset_name fraggraph2seq \
--tags lstm fraggraph2seq code64 \
--checkpoint_path ../resource/checkpoint/lstm_fraggraph2seq_code64
