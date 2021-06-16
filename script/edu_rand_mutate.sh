#!/bin/bash

python pretrain_edu.py --randomize_smiles --mutate --checkpoint_tag rand_mutate
python hillclimb.py --checkpoint_tag rand_mutate
python hillclimb.py --checkpoint_tag rand_mutate --goal_lr 1e-1
