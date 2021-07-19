#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="ae_dgi"

python train_ae.py \
--ae_type dgi_ae \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

for i in 1 2 3 4 5
do
    python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
done