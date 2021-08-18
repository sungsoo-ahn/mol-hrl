#!/bin/bash

python run_autoencoder.py \
--decoder_name transformer_base \
--tags no_vq transformer graph2seq \
--checkpoint_path ../resource/checkpoint/no_vq_transformer_graph2seq