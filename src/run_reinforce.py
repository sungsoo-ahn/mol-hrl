import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from model.reinforce import ReinforceModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ReinforceModel.add_args(parser)
    parser.add_argument("--decoder_save_path", type=str, default="")
    args = parser.parse_args()

    model = ReinforceModel(
        args.encoder_num_layer,
        args.encoder_emb_dim,
        args.encoder_load_path,
        args.encoder_optimize,
        args.decoder_num_layers,
        args.decoder_hidden_dim,
        args.code_dim,
        args.decoder_load_path,
        args.data_dir,
        args.batch_size,
        args.batches_per_epoch,
        args.reward_temperature,
        )
    
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", 
        experiment_name="neptune_logs", 
        params=vars(args),
    )
    
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=50,
    )
    trainer.fit(model)

    if args.decoder_save_path != "":
        torch.save(model.decoder.state_dict(), args.decoder_save_path)
