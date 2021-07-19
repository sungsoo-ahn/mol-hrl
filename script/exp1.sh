#!/bin/bash

CHECKPOINT_DIR="../resource/checkpoint"
TAG="ae_mask"

python train_ae.py \
--ae_type ae \
--input_graph_transform_type mask \
--checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
--tag $TAG

for i in 1 2 3 4 5
do
    python eval_ae.py --checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" --tag $TAG
done