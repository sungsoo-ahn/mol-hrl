#!/bin/bash

#python run_autoencoder.py \
#--vq \
#--code_dim 128 \
#--dataset_name fraggraph2seq \
#--tags vq lstm fraggraph2seq \
#--checkpoint_path ../resource/checkpoint/vq_lstm_fraggraph2seq 

python run_plug.py \
--vq \
--code_dim 128 \
--load_checkpoint_path "../resource/checkpoint/vq_lstm_fraggraph2seq/best.ckpt"