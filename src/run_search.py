import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from data.datamodule import SmilesDataModule
from model.search import SearchModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    SearchModel.add_args(parser)
    args = parser.parse_args()

    model = SearchModel(
        args.encoder_num_layer,
        args.encoder_emb_dim,
        args.encoder_load_path,
        args.encoder_optimize,
        args.decoder_num_layers,
        args.decoder_hidden_dim,
        args.decoder_code_dim,
        args.decoder_load_path,
        args.decoder_optimize,
        args.buffer_capacity,
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.num_warmup_samples,
        args.scoring_name,
        args.queries_per_epoch,
        args.sample_batch_size, 
        args.batches_per_epoch,
        )
    
    """
    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", 
        experiment_name="neptune-imitation", 
        params=vars(args),
    )
    """

    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=50,
    )
    trainer.fit(model)