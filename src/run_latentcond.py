import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.backbone import BackBoneModel
from model.sl_encoder import SupervisedLearningEncoderModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BackBoneModel.add_args(parser)
    SupervisedLearningEncoderModel.add_args(parser)
    parser.add_argument("--encoder_save_path", type=str, default="")
    hparams = parser.parse_args()

    backbone = BackBoneModel(hparams)
    sl_encoder = SupervisedLearningEncoderModel(backbone, hparams)
    
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", 
        experiment_name="neptune_logs", 
        params=vars(hparams),
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='sl_encoder/validation/loss/total',
        save_top_k=1,
        )

    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=50,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(sl_encoder)
    sl_encoder.load_from_checkpoint(checkpoint_callback.best_model_path)

    

    """
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=50,
    )
    trainer.fit(sl_encoder)
    """