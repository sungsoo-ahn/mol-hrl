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
    parser.add_argument("--max_freeze_epochs", type=int, default=1000)
    parser.add_argument("--max_melt_epochs", type=int, default=1000)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default_codedecoder.pth")
    parser.add_argument("--tags", type=str, nargs="+", default=[])
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", experiment_name="run_conddecoder", params=vars(hparams),
    )
    neptune_logger.append_tags(["conddecoder"] + hparams.tags)

    model = CondDecoderModule(hparams)
    
    model.freeze_decoder = True
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_freeze_epochs,
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)

    model.freeze_decoder = False
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_melt_epochs,
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)


    state_dict = {"decoder": model.decoder.state_dict(), "cond_embedding": model.cond_embedding.state_dict()}
    torch.save(state_dict, hparams.checkpoint_path)
