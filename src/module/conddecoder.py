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

        if hparams.load_checkpoint_path != "":
            state_dict = torch.load(hparams.load_checkpoint_path)
            if "decoder" in state_dict:
                self.decoder.load_state_dict(state_dict["decoder"])
            elif "cond_embedding" in state_dict:
                self.cond_embedding.load_state_dict(state_dict["cond_embedding"])

        self.train_cond_dataset = ScoreDataset(hparams.data_dir, hparams.score_func_name, hparams.train_split)
        self.train_target_dataset = SequenceDataset(hparams.data_dir, hparams.train_split)
        self.train_dataset = ZipDataset(self.train_cond_dataset, self.train_target_dataset)
        
        _, self.score_func, self.corrupt_score = get_scoring_func(hparams.score_func_name)
        
        self.freeze_decoder = True

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--sample_eval_freq", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--cond_embedding_mlp", action="store_true")

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--load_checkpoint_path", type=str, default="")
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_queries", type=int, default=10000)
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
            drop_last=True
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
        self.log("train/loss/total", loss, on_step=False, on_epoch=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.evaluate_per_n_epoch == 0:
            self.eval()
            with torch.no_grad():
                self.evaluate_sampling()
                
            self.train()

    def evaluate_sampling(self):
        if self.hparams.score_func_name == "penalized_logp":
            score_queries = [4.0, 5.0, 6.0, 7.0]
            success_margin = 0.2
        elif self.hparams.score_func_name == "logp":
            score_queries = [4.0, 5.0, 6.0, 7.0]
            success_margin = 0.2
        elif self.hparams.score_func_name == "molwt":
            score_queries = [500.0, 600.0, 700.0, 800.0]
            success_margin = 20.0
        elif self.hparams.score_func_name == "qed":
            score_queries = [0.75, 0.8, 0.85, 0.9]
            success_margin = 0.01

        for query in score_queries:
            smiles_list = []
            for _ in range(self.hparams.num_queries // self.hparams.query_batch_size):
                query_tsr = torch.full((self.hparams.query_batch_size, 1), query, device=self.device)
                batched_cond_data = self.train_cond_dataset.normalize(query_tsr)
                codes = self.cond_embedding(batched_cond_data)
                smiles_list_ = self.decoder.sample_smiles(codes, argmax=True)
                smiles_list.extend(smiles_list_)
            
            smiles_list = smiles_list[:self.hparams.num_queries]
            unique_smiles_list = list(set(smiles_list))

            unique_ratio = float(len(unique_smiles_list)) / len(smiles_list)
            self.log(f"query{query:.2f}/unique_ratio", unique_ratio, on_step=False, logger=True)

            score_list = self.score_func(smiles_list)
            valid_idxs = [idx for idx, score in enumerate(score_list) if score > self.corrupt_score]
            valid_scores = [score_list[idx] for idx in valid_idxs]

            valid_ratio = len(valid_scores) / len(score_list)
            self.log(f"query{query:.2f}/valid_ratio", valid_ratio, on_step=False, logger=True)

            is_success = lambda score: (score > query - success_margin) and (score < query + success_margin)
            success_ratio = float(len([score for score in valid_scores if is_success(score)])) / len(smiles_list)
            self.log(f"query{query:.2f}/success_ratio", success_ratio, on_step=False, logger=True)
            
            if valid_ratio > 0.0:
                valid_scores_tsr = torch.FloatTensor(valid_scores)
                mae = (query - valid_scores_tsr).abs().mean()
                self.log(f"query{query:.2f}/mae_score", mae, on_step=False, logger=True)

                mean = valid_scores_tsr.mean()
                self.log(f"query{query:.2f}/mean_score", mean, on_step=False, logger=True)

                std = valid_scores_tsr.std() if len(valid_scores) > 1 else 0.0
                self.log(f"query{query:.2f}/std_score", std, on_step=False, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{"params": self.cond_embedding.parameters()}, {"params": self.decoder.parameters()}], lr=self.hparams.lr
            )
        lambda0 = lambda epoch: 0.0 if epoch < self.hparams.max_decoder_freeze_epochs else 1.0
        lambda1 = lambda epoch: 10.0 if epoch < self.hparams.max_decoder_freeze_epochs else 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0, lambda1])
        return [optimizer], [scheduler]
