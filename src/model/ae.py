from argparse import Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from net.rnn import RnnDecoder, compute_sequence_accuracy, compute_sequence_cross_entropy
from net.gnn import GnnEncoder
from data.sequence.dataset import SequenceDataset
from data.sequence.collate import collate_sequence_data_list
from data.sequence.util import string_from_sequence
from data.pyg.dataset import PyGDataset
from data.pyg.collate import collate_pyg_data_list
from data.pyg.util import pyg_from_string
from data.score.dataset import ScoresDataset
from data.util import ZipDataset, load_raw_data


def collate_data_list(data_list):
    sequence_data_list, pyg_data_list = zip(*data_list)
    return (
        collate_sequence_data_list(sequence_data_list, pad_id=0),
        collate_pyg_data_list(pyg_data_list),
    )

def collate_aug_data_list(data_list):
    sequence_data_list, pyg_data_list, pyg_aug_data_list = zip(*data_list)
    return (
        collate_sequence_data_list(sequence_data_list, pad_id=0),
        collate_pyg_data_list(pyg_data_list),
        collate_pyg_data_list(pyg_aug_data_list)
    )

class AutoEncoderModel(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModel, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        
        self.contrastive_coef = hparams.contrastive_coef
        self.setup_base_datasets(hparams)
        self.setup_base_models(hparams)

    def setup_base_datasets(self, hparams):
        self.datasets = dict()
        smiles_list, _, split_idxs = load_raw_data(
            hparams.data_dir, hparams.score_func_names, hparams.train_ratio, hparams.label_ratio,
        )
        self.datasets["full/smiles"] = smiles_list
        self.datasets["full/sequence"] = SequenceDataset(smiles_list)
        self.tokenizer = self.datasets["full/sequence"].tokenizer
        self.vocabulary = self.datasets["full/sequence"].vocabulary
        self.datasets["full/pyg"] = PyGDataset(smiles_list)
        self.datasets["full/pyg_aug"] = PyGDataset(smiles_list, mutate=True)
        
        for split_key in ["train", "val"]:
            for data_key in ["smiles", "sequence", "pyg", "pyg_aug"]:
                self.datasets[f"{split_key}/{data_key}"] = Subset(
                    self.datasets[f"full/{data_key}"], split_idxs[split_key]
                )

        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size
        self.contrastive_coef = hparams.contrastive_coef
        if self.contrastive_coef > 0.0:
            self.train_dataset = ZipDataset(
                [
                    self.datasets["train/sequence"],
                    self.datasets["train/pyg"],
                    self.datasets["train/pyg_aug"],
                ]
            )
            self.val_dataset = ZipDataset(
                [
                    self.datasets["val/sequence"],
                    self.datasets["val/pyg"],
                    self.datasets["val/pyg_aug"],
                ]
            )
        else:
            self.train_dataset = ZipDataset(
                [self.datasets["train/sequence"], self.datasets["train/pyg"]]
                )
        
            self.val_dataset = ZipDataset([self.datasets["val/sequence"], self.datasets["val/pyg"]])

    def setup_base_models(self, hparams):
        vocab_size = len(self.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=hparams.decoder_num_layers,
            input_dim=vocab_size,
            output_dim=vocab_size,
            hidden_dim=hparams.decoder_hidden_dim,
            code_dim=hparams.code_dim,
            spherical=hparams.spherical,
        )
        self.encoder = GnnEncoder(
            num_layer=hparams.encoder_num_layer,
            emb_dim=hparams.encoder_emb_dim,
            code_dim=hparams.code_dim,
            spherical=hparams.spherical,
        )
        

    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--train_ratio", type=float, default=0.9)
        parser.add_argument("--label_ratio", type=float, default=0.1)
        parser.add_argument(
            "--score_func_names",
            type=str,
            nargs="+",
            default=["penalized_logp", "molwt", "qed", "tpsa"],
        )
        
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=24)

        parser.add_argument("--code_dim", type=int, default=32)
        parser.add_argument("--encoder_num_layer", type=int, default=5)
        parser.add_argument("--encoder_emb_dim", type=int, default=256)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--spherical", action="store_true")

        parser.add_argument("--contrastive_coef", type=float, default=0.0)

        return parser

    def train_dataloader(self):
        collate_fn = collate_aug_data_list if self.contrastive_coef > 0 else collate_data_list
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        collate_fn = collate_aug_data_list if self.contrastive_coef > 0 else collate_data_list
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def training_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data)

        self.log("train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data)

        self.log(
            "validation/loss/total", loss_total, on_step=False, logger=True,
        )
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]

    def shared_step(self, batched_data):
        if self.contrastive_coef > 0.0:
            batched_sequence_data, batched_pyg_data, batched_pyg_aug_data = batched_data
        else:
            batched_sequence_data, batched_pyg_data = batched_data
        
        codes = self.encoder(batched_pyg_data)
        if self.contrastive_coef > 0.0:
            codes_aug = self.encoder(batched_pyg_aug_data)
        
        logits = self.decoder(batched_sequence_data, codes)

        loss_ce = self.compute_cross_entropy(logits, batched_sequence_data)
        acc_elem, acc_sequence = self.compute_accuracy(logits, batched_sequence_data)

        loss = loss_ce
        statistics = {
            "loss/ce": loss_ce,
            "acc/elem": acc_elem,
            "acc/sequence": acc_sequence,
        }
        
        if self.contrastive_coef > 0.0:
            contrastive_logits = torch.mm(codes, codes_aug.T) 
            contrastive_labels = torch.arange(codes.size(0)).to(self.device)
            loss_contrastive = F.cross_entropy(contrastive_logits, contrastive_labels)
            loss += self.contrastive_coef * loss_contrastive
            acc_contrastive = (
                (torch.argmax(contrastive_logits, dim=-1) == contrastive_labels).float().mean()
            )

            statistics["loss/contrastive"] = loss_contrastive
            statistics["acc/contrastive"] = acc_contrastive    

        return loss, statistics
        
    # Wrapper functions
    def compute_cross_entropy(self, logits, batched_sequence_data):
        return compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
        )

    def compute_accuracy(self, logits, batched_sequence_data):
        return compute_sequence_accuracy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
        )

    def encode(self, smiles_list):
        pyg_data_list = [pyg_from_string(smiles) for smiles in smiles_list]
        batched_pyg_data = collate_pyg_data_list(pyg_data_list)
        batched_pyg_data.to(self.device)
        codes = self.encoder(batched_pyg_data)
        return codes

    def decode(self, codes):
        sequences, lengths, _ = self.decoder.argmax_sample(
            codes,
            self.vocabulary.get_start_id(),
            self.vocabulary.get_end_id(),
            self.vocabulary.get_max_length(),
        )

        sequences = sequences.cpu().split(1, dim=0)
        lengths = lengths.cpu()
        sequences = [sequence[:length] for sequence, length in zip(sequences, lengths)]

        smiles_list = [
            string_from_sequence(sequence, self.tokenizer, self.vocabulary)
            for sequence in sequences
        ]

        return smiles_list