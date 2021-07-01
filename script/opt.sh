#!/bin/bash

python opt_ae.py --load_path ../resource/log/neptune_logs/HRL-417/checkpoints/epoch=30-step=27186.ckpt --model_type vae
python opt_ae.py --load_path ../resource/log/neptune_logs/HRL-418/checkpoints/epoch=28-step=25432.ckpt --model_type vae
python opt_ae.py --load_path ../resource/log/neptune_logs/HRL-419/checkpoints/epoch=42-step=37710.ckpt --model_type vae
python opt_ae.py --load_path ../resource/log/neptune_logs/HRL-420/checkpoints/epoch=44-step=39464.ckpt --model_type ae
python opt_ae.py --load_path ../resource/log/neptune_logs/HRL-421/checkpoints/epoch=70-step=62266.ckpt --model_type ae