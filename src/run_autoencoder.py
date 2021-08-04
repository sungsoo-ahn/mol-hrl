import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from module.autoencoder import AutoEncoderModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AutoEncoderModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default.pth")
    parser.add_argument("--tags", type=str, nargs="+")
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", experiment_name="run_autoencoder", params=vars(hparams),
    )
    neptune_logger.append_tags(["autoencoder"] + hparams.tags)

    model = AutoEncoderModule(hparams)

    checkpoint_callback = ModelCheckpoint(monitor="train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)
    model.load_from_checkpoint(checkpoint_callback.best_model_path)

    state_dict = {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict()}
    torch.save(state_dict, hparams.checkpoint_path)
