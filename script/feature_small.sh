#!/bin/bash

python run_feature.py \
--logger_use_neptune \
--pretrain_tag feature \
--pretrain_epochs 200 \
--data_aug_randomize_smiles \
--data_aug_mutate \
--hillclimb_steps 100 \
--data_tag zinc_small