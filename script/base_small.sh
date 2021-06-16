#!/bin/bash

python run_base.py \
--logger_use_neptune \
--pretrain_tag base \
--data_aug_randomize_smiles \
--data_aug_mutate \
--hillclimb_steps 100 \
--data_tag zinc_small