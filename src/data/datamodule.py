import os
import torch
import pytorch_lightning as pl

from data.sequence.dataset import SequenceDataset
from data.sequence.handler import SequenceHandler
from data.pyg.dataset import PyGDataset
from data.pyg.handler import PyGHandler
from data.util import ZipDataset, get_pseudorandom_split_idxs
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("data")
        group.add_argument("--raw_dir", type=str, default="../resource/data/zinc/raw")
        group.add_argument("--batch_size", type=int, default=256)
        group.add_argument("--num_workers", type=int, default=8)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vali_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.vali_dataset.collate_fn,
            num_workers=self.num_workers,
        )

    def load_smiles_list(self, raw_dir):
        smiles_list_path = os.path.join(raw_dir, "smiles_list.txt")
        train_idxs_path = os.path.join(raw_dir, "train_idxs.pth")
        vali_idxs_path = os.path.join(raw_dir, "vali_idxs.pth")

        with open(smiles_list_path, "r") as f:
            smiles_list = f.read().splitlines()

        if not os.path.exists(train_idxs_path) or not os.path.exists(vali_idxs_path):
            train_idxs, vali_idxs = get_pseudorandom_split_idxs(len(smiles_list), [0.95, 0.05])
            torch.save(train_idxs, train_idxs_path)
            torch.save(vali_idxs, vali_idxs_path)

        with open(train_idxs_path, "r") as f:
            train_idxs = torch.load(train_idxs_path)

        with open(vali_idxs_path, "r") as f:
            vali_idxs = torch.load(vali_idxs_path)

        train_smiles_list = [smiles_list[idx] for idx in train_idxs]
        vali_smiles_list = [smiles_list[idx] for idx in vali_idxs]

        return train_smiles_list, vali_smiles_list


class SequenceDataModule(BaseDataModule):
    def __init__(self, raw_dir, batch_size, num_workers):
        super(SequenceDataModule, self).__init__()
        train_smiles_list, vali_smiles_list = self.load_smiles_list(raw_dir)
        self.handler = SequenceHandler(train_smiles_list + vali_smiles_list)
        self.train_dataset = SequenceDataset(train_smiles_list, self.handler)
        self.vali_dataset = SequenceDataset(vali_smiles_list, self.handler)

        self.batch_size = batch_size
        self.num_workers = num_workers


class PyGDataModule(BaseDataModule):
    def __init__(self, raw_dir, batch_size, num_workers):
        super(PyGDataModule, self).__init__()
        train_smiles_list, vali_smiles_list = self.load_smiles_list(raw_dir)
        self.handler = PyGHandler(train_smiles_list + vali_smiles_list)
        self.train_dataset = PyGDataset(train_smiles_list, self.handler)
        self.vali_dataset = PyGDataset(vali_smiles_list, self.handler)

        self.batch_size = batch_size
        self.num_workers = num_workers


class SequencePyGDataModule(BaseDataModule):
    def __init__(self, raw_dir, batch_size, num_workers):
        super(SequencePyGDataModule, self).__init__()
        train_smiles_list, vali_smiles_list = self.load_smiles_list(raw_dir)
        self.sequence_handler = SequenceHandler(train_smiles_list + vali_smiles_list)
        self.pyg_handler = PyGHandler()
        self.train_dataset = ZipDataset(
            SequenceDataset(train_smiles_list, self.sequence_handler),
            PyGDataset(train_smiles_list, self.pyg_handler),
        )
        self.vali_dataset = ZipDataset(
            SequenceDataset(vali_smiles_list, self.sequence_handler),
            PyGDataset(vali_smiles_list, self.pyg_handler),
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
