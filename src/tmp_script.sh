#!/bin/bash

python run_condopt.py --load_checkpoint_path ../resource/checkpoint/default_codedecoder.pth --reweight_k 1e-3
python run_condopt.py --load_checkpoint_path ../resource/checkpoint/default_codedecoder.pth --reweight_k 1e-2
python run_condopt.py --load_checkpoint_path ../resource/checkpoint/default_codedecoder.pth --reweight_k 1e-4
python run_condopt.py --load_checkpoint_path ../resource/checkpoint/default_codedecoder.pth --reweight_k 1e-5