#!/bin/bash

python run_autoencoder.py \
--vq \
--code_dim 16 \
--vq_num_vocabs 64 \
--dataset_name fraggraph2seq \
--tags vq lstm fraggraph2seq code16 vocab64 \
--checkpoint_path ../resource/checkpoint/vq_lstm_fraggraph2seq_code16_vocab64
