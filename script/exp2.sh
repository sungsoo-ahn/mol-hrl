#!/bin/bash

python run_autoencoder.py \
--code_dim 256 \
--dataset_name graph2seq \
--tags lstm graph2seq code256 \
--checkpoint_path ../resource/checkpoint/lstm_graph2seq_code256
