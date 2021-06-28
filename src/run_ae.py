import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.backbone import BackboneModel
from model.ae import AutoEncoderModel
from model.finetune import FinetuneModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BackboneModel.add_args(parser)
    AutoEncoderModel.add_args(parser)
    FinetuneModel.add_args(parser)
    hparams = parser.parse_args()

    backbone = BackboneModel(hparams)
    
    ae_model = AutoEncoderModel(backbone, hparams)
    
    score_func_name = "penalized_logp"
    queries = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    finetune_model = FinetuneModel(backbone, score_func_name, queries, hparams)

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )

    trainer = pl.Trainer(
        gpus=1, logger=neptune_logger, default_root_dir="../resource/log/", max_epochs=1,
    )
    trainer.fit(ae_model)


    trainer = pl.Trainer(
        gpus=1, logger=neptune_logger, default_root_dir="../resource/log/", max_epochs=100,
    )
    trainer.fit(finetune_model)
