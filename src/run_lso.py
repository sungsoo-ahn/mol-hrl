import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from lso.opt_module import LatentOptimizationModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    LatentOptimizationModule.add_args(parser)
    parser.add_argument("--scoring_func_names", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=100)
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )
    
    model = LatentOptimizationModule(hparams)
    
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model)
    