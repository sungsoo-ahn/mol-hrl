import os
import argparse

import pytorch_lightning as pl

import neptune.new as neptune

from module.pl_autoencoder import AutoEncoderModule
from module.pl_decoupled_autoencoder import DecoupledAutoEncoderModule
from lso.gradopt import run_gradopt
from lso.bo import run_bo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--tag", type=str, default="notag")
    args = parser.parse_args()

    try:
        model = AutoEncoderModule.load_from_checkpoint(args.checkpoint_path)
    except:
        model = DecoupledAutoEncoderModule.load_from_checkpoint(args.checkpoint_path)

    model = model.cuda()
    
    run = neptune.init(
        project="sungsahn0215/molrep",
        name="eval_ae",
        source_files=["*.py", "**/*.py"],
        tags=args.tag,
    )
    run["tag"] = args.tag
    run["log_dir"] = log_dir = f"../resource/log/{args.tag}"
    os.makedirs(log_dir, exist_ok=True)
    
    for scoring_func_name in ["penalized_logp"]:
        #run_gradopt(
        #    model, 
        #    "linear", 
        #    scoring_func_name, 
        #    run
        #    )
        #run_gradopt(
        #    model, 
        #    "gp", 
        #    scoring_func_name, 
        #    run
        #    )
        
        run_bo(model, scoring_func_name, run)