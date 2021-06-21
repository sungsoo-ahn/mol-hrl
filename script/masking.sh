#!/bin/bash

python run_imitation.py \
--encoder_load_path ../resource/checkpoint/encoder/supervised_masking.pth \
--decoder_save_path ../resource/checkpoint/decoder/imitation_supervised_masking.pth
