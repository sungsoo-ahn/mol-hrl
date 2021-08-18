#!/bin/bash

python run_autoencoder.py \
--decoder_name transformer_base \
--vq \
--code_dim 128 \
--dataset_name graph2seq \
--tags vq transformer graph2seq \
--checkpoint_path ../resource/checkpoint/vq_transformer_graph2seq