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
from module.decoder.lstm import LSTMDecoder
from pl_module.autoencoder import compute_sequence_accuracy, compute_sequence_cross_entropy

class CondDecoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CondDecoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_layers(hparams)
        self.setup_datasets(hparams)
    
    def setup_layers(self, hparams):
        self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=hparams.code_dim
            )
        self.cond_embedding = torch.nn.Linear(1, hparams.code_dim)

    def setup_datasets(self, hparams):
        self.tokenizer = load_tokenizer()
        self.scorer = load_scorer(hparams.task)        
        self.score_mean, self.score_std, self.score_success_margin = load_statistics(hparams.task)

        def build_dataset(task, split):
            input_dataset = GraphDataset(task, split)
            target_dataset = SequenceDataset(task, split)
            score_dataset = ScoreDataset(task, split)
            dataset = ZipDataset(input_dataset, target_dataset, score_dataset)
            return dataset
    
        self.train_dataset = build_dataset(hparams.task, "train")
        self.val_dataset = build_dataset(hparams.task, "valid")
        
        def collate(data_list):
            input_data_list, target_data_list, cond_data_list = zip(*data_list)
            batched_input_data = GraphDataset.collate(input_data_list)
            batched_target_data = SequenceDataset.collate(target_data_list)
            batched_cond_data = ScoreDataset.collate(cond_data_list)
            
            return batched_input_data, batched_target_data, batched_cond_data
        
        self.collate = collate

    
    @staticmethod
    def add_args(parser):
        # Common - data
        parser.add_argument("--dataset_name", type=str, default="plogp")
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--split", type=str, default="none")
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model
        parser.add_argument("--code_dim", type=int, default=256)
        parser.add_argument("--decoder_num_layers", type=int, default=2)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # training
        parser.add_argument("--lr", type=float, default=1e-3)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=5000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
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
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, on_epoch=False, logger=True)

        return loss
    
    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def evaluate_sampling(self):
        score_queries = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        for query in tqdm(score_queries):
            smiles_list = self.decode_many_smiles(
                query, num_samples=self.hparams.num_queries, max_len=self.hparams.max_len
                )
            
            statistics = self.scorer(smiles_list, query=query)
            for key, val in statistics.items():
                self.logger.run[f"query/{key}"].log(val)
    
    def decode_many_smiles(self, query, num_samples, max_len):
        offset = 0
        smiles_list = []
        for _ in range(num_samples // self.hparams.query_batch_size):
            batch_size = min(self.hparams.query_batch_size, num_samples - offset)
            offset += batch_size
            smiles_list += self.decode_smiles(query, batch_size, max_len)
        
        return smiles_list

    def decode(self, query, num_samples, max_len):
        with torch.no_grad():
            query_tsr = torch.full((num_samples, 1), query, device=self.device)
            batched_cond_data = (query_tsr - self.score_mean) / self.score_std
            codes = self.cond_embedding(batched_cond_data)
            batched_sequence_data = self.decoder.decode(codes, argmax=False, max_len=max_len)
        
        return batched_sequence_data
    
    def decode_smiles(self, query, num_samples, max_len):
        batched_sequence_data = self.decode(query, num_samples, max_len)
        smiles_list = [
            self.tokenizer.decode(data).replace(" ", "") for data in batched_sequence_data.tolist()
            ]
        return smiles_list

            
    def configure_optimizers(self):
        params = list(self.cond_embedding.parameters())
        params += list(self.decoder.parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return [optimizer]
