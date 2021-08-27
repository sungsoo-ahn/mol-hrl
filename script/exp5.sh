#!/bin/bash

python run_autoencoder.py \
--code_dim 256 \
--dataset_name fraggraph2seq \
--tags lstm fraggraph2seq code256 \
--checkpoint_path ../resource/checkpoint/lstm_fraggraph2seq_code256
