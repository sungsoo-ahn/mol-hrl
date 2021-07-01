import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.ae import AutoEncoderModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AutoEncoderModel.add_args(parser)
    hparams = parser.parse_args()
    model = AutoEncoderModel(hparams)

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    )

    checkpoint_callback = ModelCheckpoint(monitor='train/loss/total')
    trainer = pl.Trainer(
        gpus=1, 
        logger=neptune_logger,
        default_root_dir="../resource/log/", 
        max_epochs=100,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model)
    
#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