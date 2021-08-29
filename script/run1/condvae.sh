#!/bin/bash

python run1_condvae.py \
--encoder_type gnn \
--decoder_type lstm \
--tag encgnn_declstm

python run1_condvae.py \
--encoder_type gnn \
--decoder_type gru \
--tag encgnn_decgru

python run1_condvae.py \
--encoder_type lstm \
--decoder_type lstm \
--tag enclstm_declstm

python run1_condvae.py \
--encoder_type gru \
--decoder_type gru \
--tag encgru_decgru