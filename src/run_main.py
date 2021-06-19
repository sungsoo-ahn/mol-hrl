import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from data.datamodule import SequencePyGDataModule
from learning.imitation import ImitationLearningModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    SequencePyGDataModule.add_args(parser)
    ImitationLearningModel.add_args(parser)
    args = parser.parse_args()

    datamodule = SequencePyGDataModule(args.raw_dir, args.batch_size, args.num_workers)
    model = ImitationLearningModel(
        args.encoder_num_layer,
        args.encoder_emb_dim,
        args.encoder_load_path,
        args.encoder_optimize,
        args.decoder_num_layers,
        args.decoder_hidden_dim,
        args.decoder_code_dim,
        datamodule.sequence_handler,
    )

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/mol-hrl", experiment_name="neptune-imitation", params=vars(args),
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=100,
        checkpoint_callback=True,
    )
    trainer.fit(model, datamodule=datamodule)
