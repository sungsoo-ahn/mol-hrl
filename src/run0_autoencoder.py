import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from pl_module.autoencoder import AutoEncoderModule

BASE_CHECKPOINT_DIR = "../resource/checkpoint/run0_autoencoder"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AutoEncoderModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(project="sungsahn0215/molrep")
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run['sys/tags'].add(["run0", "autoencoder"] + hparams.tag.split("_"))

    model = AutoEncoderModule(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(BASE_CHECKPOINT_DIR, hparams.tag),
        monitor="validation/acc/seq",
        filename="best",
        mode="max"
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