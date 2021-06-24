import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.representation import RepresentationLearningModel, RepresentationDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    RepresentationDataModule.add_args(parser)
    RepresentationLearningModel.add_args(parser)
    parser.add_argument("--encoder_save_path", type=str, default="")
    args = parser.parse_args()

    datamodule = RepresentationDataModule(args.data_dir, args.batch_size, args.num_workers)
    model = RepresentationLearningModel(
        args.encoder_num_layer,
        args.encoder_emb_dim,
        args.code_dim
    )

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", 
        experiment_name="neptune_logs", 
        params=vars(args),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/loss/total',
        save_top_k=1,
        )

    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=50,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)
    
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    if args.encoder_save_path != "":
        torch.save(model.encoder.state_dict(), args.encoder_save_path)
