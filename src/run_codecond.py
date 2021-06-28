import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.backbone import BackboneModel
from model.selfsup import SelfSupervisedModel
from model.selfsupimitation import SelfSupervisedImitationModel
from model.imitation import ImitationModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BackboneModel.add_args(parser)
    SelfSupervisedModel.add_args(parser)
    SelfSupervisedImitationModel.add_args(parser)
    ImitationModel.add_args(parser)
    hparams = parser.parse_args()

    backbone = BackboneModel(hparams)
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl",
        experiment_name="neptune_logs",
        params=vars(hparams),
    )

    ###
    selfsup_model = SelfSupervisedModel(backbone, hparams)
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=100,
    )
    trainer.fit(selfsup_model)

    ###
    selfsupimitation_model = SelfSupervisedImitationModel(backbone, hparams)
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=50,
    )
    trainer.fit(selfsupimitation_model)

    ###
    imitation_model = ImitationModel(backbone, hparams)
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=100,
    )
    trainer.fit(imitation_model)
