#!/bin/bash

CHECKPOINT_MODEL_PATH=../resource/checkpoint/run0_autoencoder/ae/best-v1.ckpt

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 2 \
--plug_hidden_dim 256 \
--plug_latent_dim 128

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 2 \
--plug_hidden_dim 1024 \
--plug_latent_dim 128

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 3 \
--plug_hidden_dim 256 \
--plug_latent_dim 128

python run1_plugvae.py \
--load_checkpoint_path $CHECKPOINT_MODEL_PATH \
--plug_num_layers 3 \
--plug_hidden_dim 1024 \
--plug_latent_dim 128