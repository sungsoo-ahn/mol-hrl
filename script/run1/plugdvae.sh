#!/bin/bash

python run1_plugdiscretevae.py \
--load_checkpoint_path ../resource/checkpoint/run0_vqautoencoder/vqae/best.ckpt \
--plug_hidden_dim 256 \
--plug_latent_dim 128

python run1_plugdiscretevae.py \
--load_checkpoint_path ../resource/checkpoint/run0_vqautoencoder/vqae/best.ckpt \
--plug_hidden_dim 1024 \
--plug_latent_dim 128

python run1_plugdiscretevae.py \
--load_checkpoint_path ../resource/checkpoint/run0_vqautoencoder/vqae/best.ckpt \
--plug_num_layers 3 \
--plug_hidden_dim 256 \
--plug_latent_dim 128

python run1_plugdiscretevae.py \
--load_checkpoint_path ../resource/checkpoint/run0_vqautoencoder/vqae/best.ckpt \
--plug_num_layers 3 \
--plug_hidden_dim 1024 \
--plug_latent_dim 128

