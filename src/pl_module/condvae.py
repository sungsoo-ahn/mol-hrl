from typing import Sequence
from tqdm import tqdm
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.util import load_tokenizer
from data.factory import load_dataset, load_collate
from data.graph.dataset import GraphDataset
from data.sequence.dataset import SequenceDataset
from data.util import ZipDataset
from data.score.dataset import ScoreDataset, load_statistics
from data.score.score import load_scorer

from module.encoder.gnn import GNNEncoder
from module.encoder.gru import GRUEncoder
from module.encoder.lstm import LSTMEncoder
from module.decoder.gru import GRUDecoder
from module.decoder.lstm import LSTMDecoder

from pl_module.autoencoder import compute_sequence_accuracy, compute_sequence_cross_entropy
from pl_module.conddecoder import CondDecoderModule

class CondVariationalAutoEncoderModule(CondDecoderModule):
    def setup_layers(self, hparams):
        self.code_dim = hparams.code_dim
        if hparams.encoder_type == "gnn":
            self.encoder = GNNEncoder(
                num_layers=hparams.encoder_gnn_num_layers, 
                hidden_dim=hparams.encoder_gnn_hidden_dim, 
                code_dim=hparams.code_dim,
            )
        elif hparams.encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                num_layers=hparams.encoder_rnn_num_layers, 
                hidden_dim=hparams.encoder_rnn_hidden_dim, 
                code_dim=hparams.code_dim,
            )
        elif hparams.encoder_type == "gru":
            self.encoder = GRUEncoder(
                num_layers=hparams.encoder_rnn_num_layers, 
                hidden_dim=hparams.encoder_rnn_hidden_dim, 
                code_dim=hparams.code_dim,
            )

        if hparams.decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                num_layers=hparams.decoder_rnn_num_layers, 
                hidden_dim=hparams.decoder_rnn_hidden_dim, 
                code_dim=hparams.code_dim + 1
                )
        elif hparams.decoder_type == "gru":
            self.decoder = GRUDecoder(
                num_layers=hparams.decoder_rnn_num_layers, 
                hidden_dim=hparams.decoder_rnn_hidden_dim, 
                code_dim=hparams.code_dim + 1
                )

        self.cond_embedding = torch.nn.Linear(1, hparams.code_dim)
        self.linear_mu = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        self.linear_logstd = torch.nn.Linear(hparams.code_dim, hparams.code_dim)

    def setup_datasets(self, hparams):
        self.tokenizer = load_tokenizer()
        self.scorer = load_scorer(hparams.task)
        self.score_mean, self.score_std, self.score_success_margin = load_statistics(hparams.task)

        def build_dataset(task, split):
            if hparams.encoder_type == "gnn":
                input_dataset = GraphDataset(task, split)
            elif hparams.encoder_type in ["lstm", "gru"]:
                input_dataset = SequenceDataset(task, split)

            target_dataset = SequenceDataset(task, split)
            score_dataset = ScoreDataset(task, split)
            dataset = ZipDataset(input_dataset, target_dataset, score_dataset)
            
            return dataset
    
        self.train_dataset = build_dataset(hparams.task, "train")
        self.val_dataset = build_dataset(hparams.task, "valid")
        
        def collate(data_list):
            input_data_list, target_data_list, cond_data_list = zip(*data_list)
            if hparams.encoder_type == "gnn":
                batched_input_data = GraphDataset.collate(input_data_list)
            elif hparams.encoder_type in ["lstm", "gru"]:
                batched_input_data = SequenceDataset.collate(input_data_list)
            
            batched_target_data = SequenceDataset.collate(target_data_list)
            batched_cond_data = ScoreDataset.collate(cond_data_list)
            
            return batched_input_data, batched_target_data, batched_cond_data
        
        self.collate = collate

    
    @staticmethod
    def add_args(parser):
        # Common - data
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model
        parser.add_argument("--code_dim", type=int, default=256)
        
        parser.add_argument("--encoder_type", type=str, default="gnn")
        parser.add_argument("--encoder_gnn_num_layers", type=int, default=5)
        parser.add_argument("--encoder_gnn_hidden_dim", type=int, default=256)
        parser.add_argument("--encoder_rnn_num_layers", type=int, default=2)
        parser.add_argument("--encoder_rnn_hidden_dim", type=int, default=512)
        
        parser.add_argument("--decoder_type", type=str, default="lstm")
        parser.add_argument("--decoder_rnn_num_layers", type=int, default=2)
        parser.add_argument("--decoder_rnn_hidden_dim", type=int, default=512)
        
        # training
        parser.add_argument("--beta", type=float, default=1e0)
        parser.add_argument("--lr", type=float, default=1e-3)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=5000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data, batched_cond_data = batched_data
        out = self.encoder(batched_input_data)

        mu = self.linear_mu(out)
        std = (self.linear_logstd(out).exp() + 1e-6)
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        
        z = q.rsample()
        code_with_cond = torch.cat([z, batched_cond_data.view(-1, 1)], dim=1)
        logits = self.decoder(batched_target_data, code_with_cond)
        
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)
        kl_loss = (q.log_prob(z) - p.log_prob(z)).mean()
        loss = recon_loss + self.hparams.beta * kl_loss
        elem_acc, seq_acc = compute_sequence_accuracy(logits, batched_target_data)

        statistics = dict()        
        statistics["loss/total"] = loss
        statistics["loss/recon"] = recon_loss
        statistics["loss/kl"] = kl_loss
        statistics["acc/elem"] = elem_acc
        statistics["acc/seq"] = seq_acc

        return loss, statistics

    def decode(self, query, num_samples, max_len):
        with torch.no_grad():
            query_tsr = torch.full((num_samples, 1), query, device=self.device)
            batched_cond_data = (query_tsr - self.score_mean) / self.score_std
            mu = torch.zeros(num_samples, self.code_dim, device=self.device)
            std = torch.ones(num_samples, self.code_dim, device=self.device)
            p = torch.distributions.Normal(mu, std)
            z = p.sample()
            code_with_cond = torch.cat([z, batched_cond_data.view(-1, 1)], dim=1)
            batched_sequence_data = self.decoder.decode(code_with_cond, argmax=False, max_len=max_len)
        
        return batched_sequence_data