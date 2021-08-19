#!/bin/bash

#python run_autoencoder.py \
#--tags no_vq lstm graph2seq \
#--checkpoint_path ../resource/checkpoint/no_vq_lstm_graph2seq

python run_plug.py \
--tags no_vq lstm graph2seq \
--load_checkpoint_path ../resource/checkpoint/no_vq_lstm_graph2seq/best.ckpt