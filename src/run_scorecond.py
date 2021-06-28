import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.backbone import BackboneModel
from model.imitation import ImitationModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BackboneModel.add_args(parser)
    ImitationModel.add_args(parser)
    hparams = parser.parse_args()

    backbone = BackboneModel(hparams)
    imitation_model = ImitationModel(backbone, hparams)

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl",
        experiment_name="neptune_logs",
        params=vars(hparams),
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=100,
    )
    trainer.fit(imitation_model)
