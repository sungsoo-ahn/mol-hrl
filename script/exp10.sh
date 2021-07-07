#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_ae_dec_random_mutate"

python train_ae.py \
--ae_type ae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_dec_random_smiles \
--use_dec_mutate \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
