from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from module.encoder.graph import GraphEncoder
from module.decoder.sequence import SequenceDecoder
from data.graph.dataset import GraphDataset
from data.graph.transform import fragment, fragment2, mutate, rgroup
from data.graph.util import smiles2graph
from data.sequence.dataset import SequenceDataset
from data.util import ZipDataset


def collate(data_list):
    input_data_list, target_data_list = zip(*data_list)
    batched_input_data = GraphDataset.collate(input_data_list)
    batched_target_data = SequenceDataset.collate(target_data_list)
    return batched_input_data, batched_target_data


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.encoder = GraphEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)

        if hparams.input_mutate:
            self.input_transform = mutate
        elif hparams.input_fragment:
            self.input_transform = fragment
        elif hparams.input_fragment2:
            self.input_transform = fragment2
        elif hparams.input_rgroup:
            self.input_transform = rgroup
        else:
            self.input_transform = smiles2graph

        train_input_dataset = GraphDataset(self.hparams.data_dir, "train", transform=self.input_transform)
        train_target_dataset = SequenceDataset(self.hparams.data_dir, "train")
        val_input_dataset = GraphDataset(self.hparams.data_dir, "val", transform=self.input_transform)
        val_target_dataset = SequenceDataset(self.hparams.data_dir, "val")
        self.train_dataset = ZipDataset(train_input_dataset, train_target_dataset)
        self.val_dataset = ZipDataset(val_input_dataset, val_target_dataset)

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-4)

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--input_mutate", action="store_true")
        parser.add_argument("--input_fragment", action="store_true")
        parser.add_argument("--input_fragment2", action="store_true")
        parser.add_argument("--input_rgroup", action="store_true")

        #
        parser.add_argument("--code_dim", type=int, default=256)

        # GraphEncoder specific
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)
        parser.add_argument("--encoder_num_layers", type=int, default=5)

        # SequentialDecoder specific
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_max_length", type=int, default=120)

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
        )

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data = batched_data
        codes = self.encoder(batched_input_data)
        decoder_out = self.decoder(batched_target_data, codes)
        recon_loss, recon_statistics = self.decoder.compute_recon_loss(decoder_out, batched_target_data)
        loss += recon_loss
        statistics.update(recon_statistics)
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
