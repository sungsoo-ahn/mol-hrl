#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2seq_ae_dec_random"

python train_ae.py \
--ae_type ae \
--encoder_type graph \
--dm_type graph2seq \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_dec_random_smiles \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
