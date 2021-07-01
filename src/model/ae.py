from argparse import Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from net.rnn import RnnDecoder, compute_sequence_accuracy, compute_sequence_cross_entropy
from net.gnn import GnnEncoder
from data.sequence.dataset import SequenceDataset
from data.sequence.collate import collate_sequence_data_list
from data.pyg.dataset import PyGDataset
from data.pyg.collate import collate_pyg_data_list
from data.score.dataset import ScoresDataset
from data.util import ZipDataset, load_raw_data

def collate_data_list(data_list):
    sequence_data_list, pyg_data_list, score_data_list = zip(*data_list)
    return (
        collate_sequence_data_list(sequence_data_list, pad_id=0),
        collate_pyg_data_list(pyg_data_list),
        torch.stack(score_data_list, dim=0)
    )  

class AutoEncoderModel(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModel, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_base_datasets(hparams)
        self.setup_base_models(hparams)

    def setup_base_datasets(self, hparams):
        self.datasets = dict()
        smiles_list, scores_list, split_idxs = load_raw_data(
            hparams.data_dir, hparams.score_func_names, hparams.train_ratio, hparams.label_ratio,
            )
        self.datasets["full/smiles"] = smiles_list
        self.datasets["full/sequence"] = SequenceDataset(
            smiles_list, randomize_smiles=hparams.randomize_smiles
            )
        self.tokenizer = self.datasets["full/sequence"].tokenizer
        self.vocabulary = self.datasets["full/sequence"].vocabulary
        self.datasets["full/pyg"] = PyGDataset(smiles_list, mutate=hparams.mutate)
        self.datasets["full/score"] = ScoresDataset(scores_list)
        
        for split_key in ["train", "val"]:
            for data_key in ["smiles", "sequence", "pyg", "score"]:
                self.datasets[f"{split_key}/{data_key}"] = Subset(
                    self.datasets[f"full/{data_key}"], split_idxs[split_key]
                )

        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size
        self.train_dataset = ZipDataset(
            [
                self.datasets["train/sequence"], 
                self.datasets["train/pyg"], 
                self.datasets["train/score"],
            ]
            )
        self.val_dataset = ZipDataset(
            [
                self.datasets["val/sequence"],
                self.datasets["val/pyg"],
                self.datasets["val/score"],
            ]
            )

    def setup_base_models(self, hparams):
        vocab_size = len(self.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=hparams.decoder_num_layers,
            input_dim=vocab_size,
            output_dim=vocab_size,
            hidden_dim=hparams.decoder_hidden_dim,
            code_dim=hparams.code_dim,
        )
        self.encoder = GnnEncoder(
            num_layer=hparams.encoder_num_layer,
            emb_dim=hparams.encoder_emb_dim,
            code_dim=hparams.code_dim,
        )
        num_score_funcs = len(hparams.score_func_names)
        self.scores_predictor = torch.nn.Linear(hparams.code_dim, num_score_funcs)
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--train_ratio", type=float, default=0.9)
        parser.add_argument("--label_ratio", type=float, default=0.1)
        parser.add_argument("--score_func_names", type=str, nargs="+", default=[
            "penalized_logp", "molwt", "qed", "tpsa"
            ])
        parser.add_argument("--randomize_smiles", action="store_true")
        parser.add_argument("--mutate", action="store_true")
        
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=24)
        
        parser.add_argument("--code_dim", type=int, default=32)
        parser.add_argument("--encoder_num_layer", type=int, default=5)
        parser.add_argument("--encoder_emb_dim", type=int, default=256)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def shared_step(self, batched_data):
        batched_sequence_data, batched_pyg_data, scores = batched_data
        codes, _ = self.encoder(batched_pyg_data)
        logits = self.decoder(batched_sequence_data, codes)
        scores_pred = self.scores_predictor(codes.detach())

        loss_ce = self.compute_cross_entropy(logits, batched_sequence_data)
        loss_mse = F.mse_loss(scores_pred, scores)
        acc_elem, acc_sequence = self.compute_accuracy(logits, batched_sequence_data)

        loss = loss_ce + loss_mse

        return (
            loss, 
            {
                "loss/ce": loss_ce, 
                "loss/mse": loss_mse, 
                "acc/elem": acc_elem, 
                "acc/sequence": acc_sequence
            },
        )
    def compute_cross_entropy(self, logits, batched_sequence_data):
        return compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
        )

    def compute_accuracy(self, logits, batched_sequence_data):
        return compute_sequence_accuracy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
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