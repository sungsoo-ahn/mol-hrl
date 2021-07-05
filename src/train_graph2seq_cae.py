import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from module.model.cae import Graph2SeqCAEModule
from module.datamodule import Graphs2SeqDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Graphs2SeqDataModule.add_args(parser)
    Graph2SeqCAEModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default="")
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )
    
    datamodule = Graphs2SeqDataModule(hparams)
    model = Graph2SeqCAEModule(hparams)
    
    checkpoint_callback = ModelCheckpoint(monitor="train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)
    
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.save_checkpoint(hparams.checkpoint_path)
    