#!/bin/bash

python train_ae.py \
--ae_type aae \
--mask_rate 0.1 \
--checkpoint_path "../resource/checkpoint/seq2seq_aae_mask.pth"