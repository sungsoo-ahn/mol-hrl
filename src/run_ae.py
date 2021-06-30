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
    
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )

    checkpoint_callback = ModelCheckpoint(monitor='autoencoder/train/loss/total')
    trainer = pl.Trainer(
        gpus=1, 
        logger=neptune_logger, 
        default_root_dir="../resource/log/", 
        max_epochs=100,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(ae_model)
    
    state_dict = ae_model.state_dict()
    torch.save(state_dict, "ae.pth")
        
    ###    
    #
    score_func_name = "molwt"
    queries = [250.0, 350.0, 450.0]
    finetune_model = FinetuneModel(backbone, score_func_name, queries, hparams)

    trainer = pl.Trainer(
        gpus=1, 
        logger=neptune_logger, 
        default_root_dir="../resource/log/", 
        max_epochs=100,
        gradient_clip_val=1.0,
    )
    trainer.fit(finetune_model)
    
    #
    score_func_name = "logp"
    queries = [1.5, 3.0, 4.5]
    finetune_model = FinetuneModel(backbone, score_func_name, queries, hparams)

    trainer = pl.Trainer(
        gpus=1, 
        logger=neptune_logger, 
        default_root_dir="../resource/log/", 
        max_epochs=100,
        gradient_clip_val=1.0,
    )
    trainer.fit(finetune_model)

    #
    score_func_name = "qed"
    queries = [0.5, 0.7, 0.9]
    finetune_model = FinetuneModel(backbone, score_func_name, queries, hparams)

    trainer = pl.Trainer(
        gpus=1, 
        logger=neptune_logger, 
        default_root_dir="../resource/log/", 
        max_epochs=100,
        gradient_clip_val=1.0,
    )
    trainer.fit(finetune_model)
