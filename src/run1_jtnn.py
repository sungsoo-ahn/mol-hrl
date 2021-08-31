import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from pl_module.jtnn import JTNNModule

BASE_CHECKPOINT_DIR = "../resource/checkpoint/run1_jtnn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    JTNNModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = JTNNModule(hparams)
    model.postsetup_datasets()
    neptune_logger = NeptuneLogger(project="sungsahn0215/molrep", close_after_fit=False)
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run['sys/tags'].add(["run1", "jtnn"] + hparams.tag.split("_"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(BASE_CHECKPOINT_DIR, hparams.tag),
        monitor="validation/loss/total",
        filename="best",
        mode="min"
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

    model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    model.eval()
    model = model.cuda()
    model.evaluate_sampling()
    