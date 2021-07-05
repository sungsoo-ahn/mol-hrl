import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from data.graph.dataset import GraphDataset
from data.seq.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
from data.util import ZipDataset, load_raw_data

class LatentRegressorDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(LatentRegressorDataModule, self).__init__()
        self.hparams = hparams
        self.setup_datasets(hparams)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dm_type", type=str, default="seq")
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--score_func_names", type=str, nargs="+", default=["penalized_logp"])
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

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

    def setup_datasets(self, hparams):
        if hparams.dm_type == "seq":
            train_input_dataset = SequenceDataset(hparams.data_dir, split="train_labeled")
            train_score_dataset = ScoreDataset(
                hparams.data_dir, score_func_names=hparams.score_func_names, split="train_labeled"
                )
            self.train_dataset = ZipDataset(train_input_dataset, train_score_dataset)

            val_input_dataset = SequenceDataset(hparams.data_dir, split="val")
            val_score_dataset = ScoreDataset(hparams.data_dir, score_func_names=hparams.score_func_names, split="val")
            self.val_dataset = ZipDataset(val_input_dataset, val_score_dataset)

        elif hparams.dm_type == "graph":
            train_input_dataset = GraphDataset(hparams.data_dir, split="train_labeled")
            train_score_dataset = ScoreDataset(hparams.data_dir, score_func_names=hparams.score_func_names, split="train_labeled")
            self.train_dataset = ZipDataset(train_input_dataset, train_score_dataset)

            val_input_dataset = GraphDataset(hparams.data_dir, split="val")
            val_score_dataset = ScoreDataset(
                hparams.data_dir, score_func_names=hparams.score_func_names, split="val"
                )
            self.val_dataset = ZipDataset(val_input_dataset, val_score_dataset)
