#!/bin/bash

python run_condopt.py \
--weighted \
--train_split train \
--num_warmup_steps 1000 \
--num_steps_per_stage 50
