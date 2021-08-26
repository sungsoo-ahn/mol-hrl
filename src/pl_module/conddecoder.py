from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.util import load_tokenizer
from data.factory import load_dataset, load_collate
from data.score.score import BindingScorer, PLogPScorer
from data.score.dataset import PLOGP_MEAN, PLOGP_STD, PLOGP_SUCCESS_MARGIN
from module.decoder.lstm import LSTMDecoder
from pl_module.autoencoder import compute_sequence_accuracy, compute_sequence_cross_entropy

class CondDecoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CondDecoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.decoder = LSTMDecoder(
            decoder_num_layers=hparams.decoder_num_layers, 
            decoder_hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=hparams.code_dim
            )
        self.cond_embedding = torch.nn.Linear(1, hparams.code_dim)

        self.dataset = load_dataset(hparams.dataset_name, hparams.task, hparams.split)
        self.collate = load_collate(hparams.dataset_name)
        
        self.tokenizer = load_tokenizer()
        if hparams.task == "plogp":
            self.scorer = PLogPScorer() 
        elif hparams.task == "binding":
            self.scorer = BindingScorer(hparams.split, "default")


    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--cond_embedding_mlp", action="store_true")

        # Common - data
        parser.add_argument("--decoder_name", type=str, default="lstm_small")
        parser.add_argument("--dataset_name", type=str, default="plogp")
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--split", type=str, default="none")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_queries", type=int, default=5000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        #
        parser.add_argument("--decoder_num_layers", type=int, default=2)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--code_dim", type=int, default=256)

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        _, batched_target_data, batched_cond_data = batched_data
        codes = self.cond_embedding(batched_cond_data.unsqueeze(1))
        logits = self.decoder(batched_target_data, codes)
        
        logits = self.decoder(batched_target_data, codes)
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)
        elem_acc, seq_acc = compute_sequence_accuracy(logits, batched_target_data)
        
        loss += recon_loss
        statistics["loss/recon"] = loss
        statistics["acc/elem"] = elem_acc
        statistics["acc/seq"] = seq_acc
        
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("train/loss/total", loss, on_step=True, on_epoch=False, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, on_epoch=False, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.evaluate_per_n_epoch == 0:
            self.eval()
            with torch.no_grad():
                self.evaluate_sampling()
            
            self.train()

    def evaluate_sampling(self):
        score_queries = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        score_mean, score_std = PLOGP_MEAN, PLOGP_STD
        success_margin = PLOGP_SUCCESS_MARGIN
        for query in score_queries:
            statistics = self.sample_queries(
                query, success_margin, score_mean, score_std, num_samples=self.hparams.num_queries
                )
            for key, val in statistics.items():
                self.log(f"query{query:.2f}/{key}", val, on_step=False, logger=True)

    def sample_queries(self, query, success_margin, score_mean, score_std, num_samples):
        smiles_list = []
        for _ in range(num_samples // self.hparams.query_batch_size):
            query_tsr = torch.full((self.hparams.query_batch_size, 1), query, device=self.device)
            batched_cond_data = (query_tsr - score_mean) / score_std
            codes = self.cond_embedding(batched_cond_data)
            
            batched_sequence_data = self.decoder.sample(codes, argmax=False, max_len=self.hparams.max_len)
            smiles_list_ = [
                self.tokenizer.decode(data).replace(" ", "") for data in batched_sequence_data.tolist()
                ]
            smiles_list.extend(smiles_list_)

        statistics = self.scorer(smiles_list, query=query, success_margin=success_margin)
        return statistics
            
            
    def configure_optimizers(self):
        params = list(self.cond_embedding.parameters())
        params += list(self.decoder.parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return [optimizer]
