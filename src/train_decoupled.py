import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from module.pl_decoupled_autoencoder import DecoupledAutoEncoderModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DecoupledAutoEncoderModule.add_args(parser)
    parser.add_argument("--stage0_max_epochs", type=int, default=100)
    parser.add_argument("--stage1_max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default.pth")
    parser.add_argument("--tag", type=str, default="notag")
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", experiment_name="train_ae", params=vars(hparams),
    )
    neptune_logger.append_tags([hparams.tag])

    model = DecoupledAutoEncoderModule(hparams)

    checkpoint_callback = ModelCheckpoint(monitor="stage0/train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.stage0_max_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

    model = DecoupledAutoEncoderModule.load_from_checkpoint(
        checkpoint_callback.best_model_path, 
        stage=1, 
        input_smiles_transform_type="none",
        input_selfie_transform_type="none",
        input_graph_transform_type="none",
        )
    
    checkpoint_callback = ModelCheckpoint(monitor="stage1/train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.stage1_max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)

    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.save_checkpoint(hparams.checkpoint_path)
