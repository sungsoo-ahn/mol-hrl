import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

# from module.autoencoder import AutoEncoderModule
from module.conddecoder import CondDecoderModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CondDecoderModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default_codedecoder.pth")
    parser.add_argument("--tag", type=str, default="notag")
    hparams = parser.parse_args()

    #if hparams.load_checkpoint_path != "":
    #    hparams.max_epochs = 20
    #    hparams.check_val_every_n_epoch = 2

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", experiment_name="run_conddecoder", params=vars(hparams),
    )
    neptune_logger.append_tags([hparams.tag])

    model = CondDecoderModule(hparams)

    checkpoint_callback = ModelCheckpoint(monitor="train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
    )
    trainer.fit(model)

    state_dict = {"decoder": model.decoder.state_dict(), "cond_embedding": model.cond_embedding.state_dict()}
    torch.save(state_dict, hparams.checkpoint_path)
