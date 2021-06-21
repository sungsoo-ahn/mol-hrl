import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from data.datamodule import SmilesDataModule
from model.imitation import ImitationLearningModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    SmilesDataModule.add_args(parser)
    ImitationLearningModel.add_args(parser)
    parser.add_argument("--encoder_save_path", type=str, default="")
    parser.add_argument("--decoder_save_path", type=str, default="")
    args = parser.parse_args()

    datamodule = SmilesDataModule(args.data_dir, args.batch_size, args.num_workers)
    model = ImitationLearningModel(
        args.encoder_num_layer,
        args.encoder_emb_dim,
        args.encoder_load_path,
        args.encoder_optimize,
        args.decoder_num_layers,
        args.decoder_hidden_dim,
        args.decoder_code_dim,
        args.data_dir
    )

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", 
        experiment_name="neptune-imitation", 
        params=vars(args),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss/total',
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
    
    if args.decoder_save_path != "":
        torch.save(model.decoder.state_dict(), args.decoder_save_path)
