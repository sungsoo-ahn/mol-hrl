#!/bin/bash

python train_ae.py \
--ae_type cae \
--mask_rate 0.1 \
--checkpoint_path "../resource/checkpoint/seq2seq_cae_mask.pth"