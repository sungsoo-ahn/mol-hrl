import argparse

import pytorch_lightning as pl

import neptune.new as neptune

from ae.module import AutoEncoderModule

from evaluation.knn import run_knn
from evaluation.median import run_median
from evaluation.lso_linear import run_lso_linear
#from evaluation.lso_gp import run_lso_gp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--tag", nargs="+", type=str, default=[])
    args = parser.parse_args()

    model = AutoEncoderModule.load_from_checkpoint(args.checkpoint_path)
    model = model.cuda()
    
    run = neptune.init(
        project="sungsahn0215/molrep",
        name="eval_ae",
        source_files=["*.py", "**/*.py"],
        tags=args.tag,
    )

    #run_knn(model, [1, 5, 10, 50], run)
    #run_median(model, run)
    for scoring_func_name in ["penalized_logp"]:
        run_lso_linear(model, scoring_func_name, run)
        #run_lso_gp(model, scoring_func_name, run)