#!/bin/bash

python pretrain_edu.py --randomize_smiles --mutate --swap --checkpoint_tag rand_mutate_swap
#python hillclimb.py --checkpoint_tag rand_mutate_swap
python hillclimb.py --checkpoint_tag rand_mutate_swap --goal_lr 1e-1
python hillclimb.py --checkpoint_tag base --goal_lr 1e-1
