from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset
from data.util import ZipDataset


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(AutoEncoderDataModule, self).__init__()
        self.hparams = hparams
        self.setup_datasets(hparams)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dm_type", type=str, default="seq2seq")
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        # sequence specific augmentation
        parser.add_argument("--use_random_smiles", action="store_true")
        parser.add_argument("--use_dec_random_smiles", action="store_true")
        parser.add_argument("--mask_rate", type=float, default=0.0)

        # mutation_specific augmentation
        parser.add_argument("--use_mutate", action="store_true")
        parser.add_argument("--use_dec_mutate", action="store_true")
        
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
        if hparams.dm_type == "seq2seq":
            train_input_dataset = SequenceDataset(
                hparams.data_dir,
                split="train",
                use_random_smiles=hparams.use_random_smiles,
                mask_rate=hparams.mask_rate,
                mutate=hparams.use_mutate,
            )
            train_target_dataset = SequenceDataset(
                hparams.data_dir, split="train", 
                use_random_smiles=hparams.use_dec_random_smiles,
                mutate=hparams.use_dec_mutate,
                )
            self.train_dataset = ZipDataset(train_input_dataset, train_target_dataset)

            val_input_dataset = SequenceDataset(hparams.data_dir, split="val")
            val_target_dataset = SequenceDataset(hparams.data_dir, split="val")
            self.val_dataset = ZipDataset(val_input_dataset, val_target_dataset)

            hparams.num_vocabs = len(train_target_dataset.vocabulary)
        elif hparams.dm_type == "seqs2seq":
            train_input_dataset0 = SequenceDataset(hparams.data_dir, split="train")
            train_input_dataset1 = SequenceDataset(
                hparams.data_dir,
                split="train",
                use_random_smiles=hparams.use_random_smiles,
                mask_rate=hparams.mask_rate,
                mutate=hparams.use_mutate,
            )
            train_input_dataset = ZipDataset(train_input_dataset0, train_input_dataset1)
            train_target_dataset = SequenceDataset(
                hparams.data_dir, split="train", 
                use_random_smiles=hparams.use_dec_random_smiles,
                mutate=hparams.use_dec_mutate,
                )
            self.train_dataset = ZipDataset(train_input_dataset, train_target_dataset)

            val_input_dataset0 = SequenceDataset(hparams.data_dir, split="val")
            val_input_dataset1 = SequenceDataset(
                hparams.data_dir,
                split="val",
                use_random_smiles=hparams.use_random_smiles,
                mask_rate=hparams.mask_rate,
                mutate=hparams.use_mutate,
            )
            val_input_dataset = ZipDataset(val_input_dataset0, val_input_dataset1)
            val_target_dataset = SequenceDataset(hparams.data_dir, split="val")
            self.val_dataset = ZipDataset(val_input_dataset, val_target_dataset)

            hparams.num_vocabs = len(train_target_dataset.vocabulary)

        elif hparams.dm_type == "graph2seq":
            train_input_dataset = GraphDataset(
                hparams.data_dir, split="train", mutate=hparams.use_mutate
            )
            train_target_dataset = SequenceDataset(
                hparams.data_dir, 
                split="train", 
                use_random_smiles=hparams.use_dec_random_smiles,
                mutate=hparams.use_dec_mutate,
                )
            self.train_dataset = ZipDataset(train_input_dataset, train_target_dataset)

            val_input_dataset = GraphDataset(hparams.data_dir, split="val")
            val_target_dataset = SequenceDataset(hparams.data_dir, split="val")
            self.val_dataset = ZipDataset(val_input_dataset, val_target_dataset)

            hparams.num_vocabs = len(train_target_dataset.vocabulary)

        elif hparams.dm_type == "graphs2seq":
            train_input_dataset0 = GraphDataset(hparams.data_dir, split="train")
            train_input_dataset1 = GraphDataset(
                hparams.data_dir, split="train", mutate=hparams.use_mutate
            )
            train_input_dataset = ZipDataset(train_input_dataset0, train_input_dataset1)
            train_target_dataset = SequenceDataset(
                hparams.data_dir, 
                split="train", 
                use_random_smiles=hparams.use_dec_random_smiles,
                mutate=hparams.use_dec_mutate,
                )
            self.train_dataset = ZipDataset(train_input_dataset, train_target_dataset)

            val_input_dataset0 = GraphDataset(hparams.data_dir, split="val")
            val_input_dataset1 = GraphDataset(
                hparams.data_dir, split="val", mutate=hparams.use_mutate
            )
            val_input_dataset = ZipDataset(val_input_dataset0, val_input_dataset1)
            val_target_dataset = SequenceDataset(hparams.data_dir, split="val")
            self.val_dataset = ZipDataset(val_input_dataset, val_target_dataset)

            hparams.num_vocabs = len(train_target_dataset.vocabulary)
