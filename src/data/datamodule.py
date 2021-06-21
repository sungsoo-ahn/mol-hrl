import os
import random
import torch
import pytorch_lightning as pl

from data.sequence.dataset import SequenceDataset
from data.sequence.handler import SequenceHandler
from data.sequence.collate import collate_sequence_data_list
from data.pyg.dataset import PyGDataset
from data.pyg.handler import PyGHandler
from data.pyg.collate import collate_pyg_data_list

from data.util import ZipDataset, load_split_smiles_list
from torch.utils.data import DataLoader

class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super(SmilesDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_smiles_list, vali_smiles_list = load_split_smiles_list(self.data_dir)
        self.sequence_handler = SequenceHandler(self.data_dir)
        self.pyg_handler = PyGHandler()
        self.train_smiles_list = train_smiles_list
        self.train_dataset = ZipDataset(
            SequenceDataset(train_smiles_list, self.sequence_handler),
            PyGDataset(train_smiles_list, self.pyg_handler),
        )
        self.vali_smiles_list = vali_smiles_list
        self.vali_dataset = ZipDataset(
            SequenceDataset(vali_smiles_list, self.sequence_handler),
            PyGDataset(vali_smiles_list, self.pyg_handler),
        )
    
    def collate_data_list(self, data_list):
        sequence_data_list, pyg_data_list = zip(*data_list)
        pad_id = self.sequence_handler.vocabulary.get_pad_id()
        return (
            collate_sequence_data_list(sequence_data_list, pad_id), 
            collate_pyg_data_list(pyg_data_list)
        )

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("data")
        group.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        group.add_argument("--batch_size", type=int, default=256)
        group.add_argument("--num_workers", type=int, default=8)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_data_list,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vali_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_data_list,
            num_workers=self.num_workers,
        )