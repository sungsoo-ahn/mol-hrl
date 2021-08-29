#!/bin/bash

#python run1_conddecoder.py \
#--decoder_num_layers 2 \
#--decoder_hidden_dim 1024 \
#--tag layer2_hidden1024

#python run1_conddecoder.py \
#--decoder_num_layers 2 \
#--decoder_hidden_dim 512 \
#--tag layer2_hidden512

#python run1_conddecoder.py \
#--decoder_num_layers 2 \
#--decoder_hidden_dim 256 \
#--tag layer2_hidden256

python run1_conddecoder.py \
--decoder_num_layers 1 \
--decoder_hidden_dim 1024 \
--tag layer1_hidden1024

python run1_conddecoder.py \
--decoder_num_layers 1 \
--decoder_hidden_dim 512 \
--tag layer1_hidden512

python run1_conddecoder.py \
--decoder_num_layers 1 \
--decoder_hidden_dim 256 \
--tag layer1_hidden256