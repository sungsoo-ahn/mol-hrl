from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from module.decoder.sequence import SequenceDecoder
from data.sequence.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
from data.score.factory import get_scoring_func
from data.util import ZipDataset


def collate(data_list):
    cond_data_list, target_data_list = zip(*data_list)
    batched_cond_data = ScoreDataset.collate(cond_data_list)
    batched_target_data = SequenceDataset.collate(target_data_list)
    return batched_cond_data, batched_target_data


class CondDecoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CondDecoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.decoder = SequenceDecoder(hparams)
        self.cond_embedding = torch.nn.Linear(1, hparams.code_dim)
        if hparams.checkpoint_path != "":
            state_dict = torch.load(hparams.checkpoint_path)
            if "decoder" in state_dict:
                self.decoder.load_state_dict(state_dict["decoder"])
            elif "cond_embedding" in state_dict:
                self.cond_embedding.load_state_dict(state_dict["cond_embedding"])

        self.train_cond_dataset = ScoreDataset(hparams.data_dir, hparams.score_func_name, hparams.train_split)
        self.train_target_dataset = SequenceDataset(hparams.data_dir, hparams.train_split)
        self.val_cond_dataset = ScoreDataset(hparams.data_dir, hparams.score_func_name, "val")
        self.val_target_dataset = SequenceDataset(hparams.data_dir, "val")
        self.train_dataset = ZipDataset(self.train_cond_dataset, self.train_target_dataset)
        self.val_dataset = ZipDataset(self.val_cond_dataset, self.val_target_dataset)

        _, self.score_func, self.corrupt_score = get_scoring_func(hparams.score_func_name)

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--cond_embedding_lr", type=float, default=1e-2)
        parser.add_argument("--decoder_lr", type=float, default=1e-4)

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--query_batch_size", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--score_func_name", type=str, default="penalized_logp")

        #
        parser.add_argument("--code_dim", type=int, default=256)

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
        batched_cond_data, batched_target_data = batched_data
        codes = self.cond_embedding(batched_cond_data)
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

    def on_validation_epoch_end(self):
        if self.hparams.score_func_name == "penalized_logp":
            score_queries = [0.0, 2.0, 4.0, 6.0]
        elif self.hparams.score_func_name == "logp":
            score_queries = [3.0, 6.0, 9.0, 12.0]
        elif self.hparams.score_func_name == "molwt":
            score_queries = [200.0, 400.0, 600.0, 800.0]
        elif self.hparams.score_func_name == "qed":
            score_queries = [0.5, 0.7, 0.9, 1.0]

        for query in score_queries:
            normalized_query = self.train_cond_dataset.normalize(query).item()
            batched_cond_data = torch.full((self.hparams.query_batch_size, 1), normalized_query, device=self.device)
            codes = self.cond_embedding(batched_cond_data)
            smiles_list = self.decoder.sample_smiles(codes, argmax=False)
            scores_list = self.score_func(smiles_list)
            valid_scores = torch.FloatTensor([score for score in scores_list if score > self.corrupt_score])

            valid_ratio = valid_scores.size(0) / len(scores_list)
            self.log(f"query{query:.2f}/valid_ratio", valid_ratio, on_step=False, logger=True)

            if valid_ratio > 0.0:
                mae = (query - valid_scores).abs().mean()
                self.log(f"query{query:.2f}/mae", mae, on_step=False, logger=True)

                max_score = valid_scores.max()
                self.log(f"query{query:.2f}/max_score", max_score, on_step=False, logger=True)

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": self.decoder.parameters(), "lr": self.hparams.decoder_lr},
            {"params": self.cond_embedding.parameters(), "lr": self.hparams.cond_embedding_lr},
            ]
        optimizer = torch.optim.Adam(grouped_parameters)
        return [optimizer]
