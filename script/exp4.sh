#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="seq2seq_ae_dec_random"

python train_ae.py \
--ae_type ae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--use_dec_random_smiles \
--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG seq