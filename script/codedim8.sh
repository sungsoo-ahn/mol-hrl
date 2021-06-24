#!/bin/bash

python run_representation.py \
--code_dim 8 \
--encoder_save_path ../resource/checkpoint/encoder/codedim8.pth

python run_imitation.py \
--code_dim 8 \
--encoder_load_path \
../resource/checkpoint/encoder/codedim8.pth \
--decoder_save_path ../resource/checkpoint/decoder/imitation_codedim8.pth

python run_selfimitation.py \
--code_dim 8 \
--encoder_load_path ../resource/checkpoint/encoder/codedim8.pth \
--decoder_load_path ../resource/checkpoint/decoder/imitation_codedim8.pth \
--decoder_save_path ../resource/checkpoint/decoder/selfimitation_codedim8.pth