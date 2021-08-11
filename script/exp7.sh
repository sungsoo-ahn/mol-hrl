#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="mutate"

#python run_plug.py \
#--train_split train_001 \
#--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_plug_001.pth" \
#--tags $TAG 001

python run_play.py \
--train_split train_001 \
--load_checkpoint_path "${CHECKPOINT_DIR}/${TAG}_plug_001.pth" \
--tags $TAG 001