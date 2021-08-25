#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 16 \
--vq_num_vocabs 64 \
--dataset_name graph2seq \
--tags vq lstm graph2seq code16 vocab64 \
--checkpoint_path ../resource/checkpoint/vq_lstm_graph2seq_code16_vocab64
