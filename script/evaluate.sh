#!/bin/bash

CHECKPOINT_DIR=$1
TAG=$2
DM_TYPE=$3

#python eval_ae.py \
#--ae_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--tag $TAG

#python train_lso.py \
#--ae_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
#--checkpoint_path "${CHECKPOINT_DIR}/${TAG}_lso.pth" \
#--dm_type $DM_TYPE \
#--tag $TAG

for NAME in "logp" "qed" "molwt" "tpsa"
do
    python eval_ae.py \
    --ae_checkpoint_path "${CHECKPOINT_DIR}/${TAG}.pth" \
    --lso_checkpoint_path "${CHECKPOINT_DIR}/${TAG}_lso.pth" \
    --scoring_func_name $NAME \
    --tag $TAG
done
