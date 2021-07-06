import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from lso.opt_module import LatentOptimizationModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    LatentOptimizationModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--tag", nargs="+", type=str, default=[])
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep",
        experiment_name="run_lso",
        params=vars(hparams),
    )
    if len(hparams.tag) > 0:
        neptune_logger.append_tags(hparams.tag)

    model = LatentOptimizationModule(hparams)

    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model)
