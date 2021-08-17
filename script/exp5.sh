#!/bin/bash

python run_autoencoder.py \
--vq \
--decoder_name transformer \
--tags vq transformer \
--checkpoint_path ../resource/checkpoint/vq_transformer