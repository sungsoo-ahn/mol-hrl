#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="dgo_ae_graph"

python train_ae.py \
--ae_type dgi_ae \
--encoder_type graph \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
