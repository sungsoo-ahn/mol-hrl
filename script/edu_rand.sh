#!/bin/bash

python pretrain_edu.py --randomize_smiles --checkpoint_tag rand
python hillclimb.py --checkpoint_tag rand
python hillclimb.py --checkpoint_tag rand --goal_lr 1e-1
