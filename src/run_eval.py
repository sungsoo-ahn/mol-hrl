import os
import argparse

import pytorch_lightning as pl

import neptune.new as neptune

from module.pl_autoencoder import AutoEncoderModule
from lso.gradopt import run_gradopt
from lso.bo import run_bo
from condgen.finetune import run_finetune

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default.pth")
    parser.add_argument("--bo_covar_module", type=str, default="rbf")
    parser.add_argument("--tag", type=str, default="notag")
    args = parser.parse_args()

    run = neptune.init(
        project="sungsahn0215/molrep",
        name="eval_ae",
        source_files=["*.py", "**/*.py"],
        tags=args.tag,
    )
    run["tag"] = args.tag
    
    for scoring_func_name in [
        "penalized_logp", 
        "logp", 
        "molwt", 
        "qed", 
        "tpsa",
        ]:
        #run_gradopt(model, "linear", scoring_func_name, run)
        run_bo(scoring_func_name, run, args.bo_covar_module)
        #run_finetune(model, scoring_func_name, run)
