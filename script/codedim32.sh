#!/bin/bash

python run_representation.py \
--code_dim 32 \
--encoder_save_path ../resource/checkpoint/encoder/codedim32.pth

python run_imitation.py \
--code_dim 32 \
--encoder_load_path ../resource/checkpoint/encoder/codedim32.pth \
--decoder_save_path ../resource/checkpoint/decoder/imitation_codedim32.pth

python run_selfimitation.py \
--code_dim 32 \
--encoder_load_path ../resource/checkpoint/encoder/codedim32.pth \
--decoder_load_path ../resource/checkpoint/decoder/imitation_codedim32.pth \
--decoder_save_path ../resource/checkpoint/decoder/selfimitation_codedim32.pth