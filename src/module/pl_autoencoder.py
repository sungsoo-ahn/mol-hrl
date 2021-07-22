from argparse import Namespace

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from module.autoencoder import (
    BaseAutoEncoder, 
    ContrastiveAutoEncoder, 
    DGIContrastiveAutoEncoder, 
    RelationalAutoEncoder, 
    DGIAutoEncoder,
    StyleAutoEncoder,
    SupervisedAutoEncoder
)
from data.util import ZipDataset

class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        
        autoencoder_class = {
            "base": BaseAutoEncoder,
            "contrastive": ContrastiveAutoEncoder,
            "relational": RelationalAutoEncoder,
            "dgi": DGIAutoEncoder,
            "dgi_contrastive": DGIContrastiveAutoEncoder,
            "style": StyleAutoEncoder,
            "supervised": SupervisedAutoEncoder,
        }[hparams.autoencoder_type]
        self.autoencoder = autoencoder_class(hparams)

        self.setup_datasets()

    def setup_datasets(self):
        self.train_input_dataset = self.autoencoder.get_input_dataset("train")
        self.train_target_dataset = self.autoencoder.get_target_dataset("train")
        self.val_input_dataset = self.autoencoder.get_input_dataset("val")
        self.val_target_dataset = self.autoencoder.get_target_dataset("val")
        self.train_dataset = ZipDataset(self.train_input_dataset, self.train_target_dataset)
        self.val_dataset = ZipDataset(self.val_input_dataset, self.val_target_dataset)

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--autoencoder_type", type=str, default="base")
        parser.add_argument("--encoder_type", type=str, default="graph")
        parser.add_argument("--decoder_type", type=str, default="smiles")
        parser.add_argument("--code_dim", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-4)

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        
        parser.add_argument("--input_smiles_transform_type", type=str, default="none")
        parser.add_argument("--input_sequence_transform_type", type=str, default="none")
        parser.add_argument("--input_graph_transform_type", type=str, default="none")
        
        parser.add_argument("--target_smiles_transform_type", type=str, default="none")
        parser.add_argument("--target_sequence_transform_type", type=str, default="none")
        parser.add_argument("--target_graph_transform_type", type=str, default="none")
        
        # GraphEncoder specific
        parser.add_argument("--graph_encoder_hidden_dim", type=int, default=256)
        parser.add_argument("--graph_encoder_num_layers", type=int, default=5)

        # SequentialEncoder specific
        parser.add_argument("--sequence_encoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--sequence_encoder_num_layers", type=int, default=3)

        # SequentialDecoder specific
        parser.add_argument("--sequence_decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--sequence_decoder_num_layers", type=int, default=3)
        parser.add_argument("--sequence_decoder_max_length", type=int, default=120)

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def shared_step(self, batched_data):
        loss, statistics = self.autoencoder.update_loss(batched_data)
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("validation/loss/total", loss, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]