#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="graph2seq_ae"

#python train_ae.py \
#--ae_type ae \
#--encoder_type graph \
#--dm_type graph2seq \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

bash ../script/evaluate.sh $CHECKPOINT_DIR $TAG graph