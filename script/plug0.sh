#!/bin/bash

python run_plug.py --load_checkpoint_path ../resource/checkpoint/vq_lstm_graph2seq_code16_vocab32/best.ckpt --plug_hidden_dim 1024 --tags graph hidden1024 

python run_plug.py --load_checkpoint_path ../resource/checkpoint/vq_lstm_graph2seq_code16_vocab32/best.ckpt --plug_hidden_dim 512 --tags graph hidden512

python run_plug.py --load_checkpoint_path ../resource/checkpoint/vq_lstm_graph2seq_code16_vocab32/best.ckpt --plug_hidden_dim 256 --tags graph hidden256

