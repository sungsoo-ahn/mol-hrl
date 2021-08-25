#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 16 \
--vq_num_vocabs 128 \
--dataset_name fraggraph2seq \
--tags vq lstm fraggraph2seq code16 vocab128 \
--checkpoint_path ../resource/checkpoint/vq_lstm_fraggraph2seq_code16_vocab128
