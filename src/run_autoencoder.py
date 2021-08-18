import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from pl_module.autoencoder import AutoEncoderModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AutoEncoderModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--tags", type=str, nargs="+", default=[])
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", 
        experiment_name="run_autoencoder", 
        params=vars(hparams),
        upload_source_files=["**/*.py", "*.py"],
    )
    neptune_logger.append_tags(["autoencoder"] + hparams.tags)

    model = AutoEncoderModule(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.checkpoint_path,
        monitor="train/loss/total"
        )
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)