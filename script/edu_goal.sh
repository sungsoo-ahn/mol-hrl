#!/bin/bash

python run_edu_goal.py \
--logger_use_neptune \
--pretrain_tag edu_goal \
--data_aug_randomize_smiles \
--data_aug_mutate \
--data_tag zinc