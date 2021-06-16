#!/bin/bash

python pretrain_edu.py --randomize_smiles --swap --checkpoint_tag rand_swap
python hillclimb.py --checkpoint_tag rand_swap
python hillclimb.py --checkpoint_tag rand_swap --goal_lr 1e-1
