#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 128 \
--dataset_name graph2seq \
--tags vq lstm graph2seq \
--checkpoint_path ../resource/checkpoint/vq_lstm_graph2seq

python run_plug.py \
--vq \
--code_dim 128 \
--load_checkpoint_path "../resource/checkpoint/vq_lstm_graph2seq/best.ckpt"