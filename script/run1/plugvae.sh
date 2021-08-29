#!/bin/bash

CHECKPOINT_MODEL_PATH=../resource/checkpoint/run0_autoencoder/ae/best-v1.ckpt

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 2 \
--plug_hidden_dim 256 \
--plug_latent_dim 128 \
--tag layer2_hidden256_latent128

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 2 \
--plug_hidden_dim 1024 \
--plug_latent_dim 128 \
--tag layer2_hidden1024_latent128

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 2 \
--plug_hidden_dim 1024 \
--plug_latent_dim 256 \
--tag layer2_hidden1024_latent256