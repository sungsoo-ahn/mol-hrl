#!/bin/bash

python pretrain.py --checkpoint_tag base
python hillclimb.py --checkpoint_tag base
python hillclimb.py --checkpoint_tag base --goal_lr 1e-1
