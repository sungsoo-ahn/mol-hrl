#!/bin/bash

python run_autoencoder.py \
--code_dim 128 \
--dataset_name graph2seq \
--tags lstm graph2seq code128 \
--checkpoint_path ../resource/checkpoint/lstm_graph2seq_code128
