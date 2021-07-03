import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.ae import AutoEncoderModel
from model.lso import LatentSpaceOptimizationModel
from model.score_predictor import ScorePredictorModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AutoEncoderModel.add_args(parser)
    ScorePredictorModel.add_args(parser)
    LatentSpaceOptimizationModel.add_args(parser)
    parser.add_argument("--checkpoint_path", type=str, default="")
    hparams = parser.parse_args()
    
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )
    
    backbone = AutoEncoderModel(hparams)
    if hparams.checkpoint_path != "":
        backbone.load_from_checkpoint(hparams.checkpoint_path)
    else:
        backbone_trainer = pl.Trainer(
            gpus=1,
            logger=neptune_logger,
            default_root_dir="../resource/log/",
            max_epochs=100,
            callbacks=[ModelCheckpoint(monitor="train/loss/total")],
        )
        backbone_trainer.fit(backbone)

    score_model = ScorePredictorModel(backbone, hparams)
    score_model_trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=100,
    )
    score_model_trainer.fit(score_model)

    for scoring_func_name in hparams.score_func_names:
        lso_model = LatentSpaceOptimizationModel(backbone, score_model, scoring_func_name, hparams)
        lso_model_trainer = pl.Trainer(
            gpus=1,
            logger=neptune_logger,
            default_root_dir="../resource/log/",
            max_epochs=100,
        )
        lso_model_trainer.fit(lso_model)   

# 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
