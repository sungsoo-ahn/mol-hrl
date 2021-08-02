#!/bin/bash

python run_condopt.py --load_checkpoint_path ../resource/checkpoint/nopretrain_full.pth --reweight_k 1e-5 --tag condopt3
