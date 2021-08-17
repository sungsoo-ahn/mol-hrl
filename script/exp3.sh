#!/bin/bash

python run_autoencoder.py \
--dataset_name graph2enumseq \
--decoder_name transformer_base \
--lr 1e-4 \
--tags graph2enumseq transformer \
--checkpoint_path ../resource/checkpoint/graph2enumseq_transformer