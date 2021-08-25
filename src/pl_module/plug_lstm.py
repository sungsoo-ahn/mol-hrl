from argparse import Namespace
from enum import unique

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl

from data.factory import load_dataset, load_collate
from data.score.score import BindingScorer, PLogPScorer
from data.score.dataset import PLOGP_MEAN, PLOGP_STD
from data.util import load_tokenizer
from data.smiles.util import canonicalize
from module.factory import load_encoder, load_decoder
from module.vq_layer import VectorQuantizeLayer
from pl_module.autoencoder import AutoEncoderModule

from torch.distributions import Categorical
import random

def compute_sequence_cross_entropy(logits, batched_sequence_data):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )

    return loss

class PlugLSTM(torch.nn.Module):
    def __init__(self, vq_num_vocabs, code_dim, plug_hidden_dim, plug_num_layers, plug_temperature):
        super(PlugLSTM, self).__init__()
        self.code_dim = code_dim
        self.vq_num_vocabs = vq_num_vocabs
        self.encoder = nn.Embedding(vq_num_vocabs+1, plug_hidden_dim)
        self.y_encoder = nn.Linear(1, plug_hidden_dim)
        self.lstm = nn.LSTM(
            plug_hidden_dim,
            plug_hidden_dim,
            batch_first=True,
            num_layers=plug_num_layers,
        )
        self.decoder = nn.Linear(plug_hidden_dim, vq_num_vocabs)
        self.temperature = plug_temperature
        
    def forward(self, x, y, teacher_forcing_ratio=0.5):
        trg = torch.full((x.size(0), ), self.vq_num_vocabs, device=x.device, dtype=torch.long)
        hidden = None
        y_encoded = self.y_encoder(y.unsqueeze(1))
        logits = []
        for t in range(self.code_dim):
            trg_encoded = self.encoder(trg)
            out = trg_encoded + y_encoded
            out, hidden = self.lstm(out.unsqueeze(1), hidden)
            out = self.decoder(out)

            if random.random() < teacher_forcing_ratio:
                trg = x[:, t]
            else:
                probs = torch.softmax(out.squeeze(1), dim=1)
                distribution = Categorical(probs=probs)
                trg = distribution.sample()

            logits.append(out)

        logits = torch.cat(logits, dim=1)
        return logits
    
    def step(self, x, y):
        logits = self(x, y)

        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))

        preds = torch.argmax(logits, dim=-1)
        correct = (preds == x)
        elem_acc = correct.float().mean()
        sequence_acc = correct.view(x.size(0), -1).all(dim=1).float().mean()

        statistics = {
            "loss/plug_recon": loss,
            "acc/plug_elem": elem_acc,
            "acc/plug_seq": sequence_acc
        }

        return loss, statistics

    def sample(self, y, argmax):
        sample_size = y.size(0)
        sequences = [torch.full((sample_size, 1), self.vq_num_vocabs, dtype=torch.long).to(y.device)]
        hidden = None
        code_encoder_out = self.y_encoder(y)
        for _ in range(self.code_dim):
            out = self.encoder(sequences[-1])
            out = out + code_encoder_out.unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            if argmax == True:
                tth_sequences = torch.argmax(logit, dim=2)
            else:
                logit /= self.temperature
                prob = torch.softmax(logit, dim=2)
                distribution = Categorical(probs=prob)
                tth_sequences = distribution.sample()

            sequences.append(tth_sequences)

        sequences = torch.cat(sequences[1:], dim=1)

        return sequences

class PlugLSTMModule(pl.LightningModule):
    def __init__(self, hparams):
        super(PlugLSTMModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.tokenizer = load_tokenizer()
        code_dim = hparams.code_dim * hparams.vq_code_dim
        self.encoder = load_encoder(hparams.encoder_name, code_dim)
        self.decoder = load_decoder(hparams.decoder_name, code_dim)
        self.plug_lstm = PlugLSTM(
            hparams.vq_num_vocabs, 
            hparams.code_dim, 
            hparams.plug_hidden_dim, 
            hparams.plug_num_layers, 
            hparams.plug_temperature
            )
        self.vq_layer = VectorQuantizeLayer(hparams.vq_code_dim, hparams.vq_num_vocabs)

        if hparams.load_checkpoint_path != "":
            pl_ae = AutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint_path)
            self.encoder.load_state_dict(pl_ae.encoder.state_dict())
            self.decoder.load_state_dict(pl_ae.decoder.state_dict())
            self.vq_layer.load_state_dict(pl_ae.vq_layer.state_dict())

        self.dataset = load_dataset(hparams.dataset_name, hparams.task, hparams.split)
        self.collate = load_collate(hparams.dataset_name)
        
        if hparams.task == "plogp":
            self.scorer = PLogPScorer() 
        elif hparams.task == "binding":
            self.scorer = BindingScorer(hparams.split, "default")

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-3)
        
        # Common - data
        parser.add_argument("--dataset_name", type=str, default="plogp")
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--split", type=str, default="none")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_queries", type=int, default=250)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        
        #
        parser.add_argument("--encoder_name", type=str, default="gnn_base")
        parser.add_argument("--decoder_name", type=str, default="lstm_base")
        parser.add_argument("--vq", action="store_true")
        parser.add_argument("--vq_code_dim", type=int, default=128)
        parser.add_argument("--vq_num_vocabs", type=int, default=256)
        parser.add_argument("--code_dim", type=int, default=256)
        parser.add_argument("--max_len", type=int, default=120)
        parser.add_argument("--load_checkpoint_path", type=str, default="")

        #
        parser.add_argument("--plug_hidden_dim", type=int, default=1024)
        parser.add_argument("--plug_num_layers", type=int, default=3)
        parser.add_argument("--plug_temperature", type=float, default=1e-1)
        
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
        batched_input_data, _, batched_cond_data = batched_data
        self.encoder.eval()
        with torch.no_grad():
            codes = self.encoder(batched_input_data)
            codes, code_idxs, _ = self.vq_layer(codes)

        loss, statistics = self.plug_lstm.step(code_idxs, batched_cond_data)

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
        score_queries = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        score_mean, score_std = PLOGP_MEAN, PLOGP_STD
        success_margin = 0.5
        for query in score_queries:
            _, statistics = self.sample_queries(
                query, success_margin, score_mean, score_std, num_samples=self.hparams.num_queries
                )
            for key, val in statistics.items():
                self.log(f"query{query:.2f}/{key}", val, on_step=False, logger=True)

    def sample_queries(self, query, success_margin, score_mean, score_std, num_samples=1e4, num_unique_samples=1e4):
        smiles_list, valid_smiles_list, unique_smiles_list = [], [], []
        while len(smiles_list) < num_samples and len(unique_smiles_list) < num_unique_samples:
            batched_cond_data = torch.full((self.hparams.query_batch_size, 1), query, device=self.device)
            batched_cond_data = (batched_cond_data - score_mean) / score_std
            codes = self.plug_lstm.sample(batched_cond_data, argmax=False)
            codes = self.vq_layer.compute_embedding(codes)
        
            batched_sequence_data = self.decoder.sample(codes, argmax=True, max_len=self.hparams.max_len)
            smiles_list_ = [
                self.tokenizer.decode(data).replace(" ", "") for data in batched_sequence_data.tolist()
                ]
            valid_smiles_list_ = [smi for smi in list(map(canonicalize, smiles_list_)) if smi is not None]

            smiles_list.extend(smiles_list_)
            valid_smiles_list.extend(valid_smiles_list_)
            unique_smiles_list = list(set(valid_smiles_list))

        valid_score_list = [score for score in self.scorer(valid_smiles_list) if score is not None]
        unique_score_list = self.scorer(unique_smiles_list)

        result = dict()
        result["smiles_list"] = smiles_list
        result["valid_smiles_list"] = valid_smiles_list
        result["unique_smiles_list"] = unique_smiles_list
        result["valid_score_list"] = valid_score_list
        result["unique_score_list"] = unique_score_list

        statistics = dict()
        statistics["valid_ratio"] = float(len(valid_smiles_list)) / len(smiles_list)
        statistics["unique_ratio"] = float(len(set(smiles_list))) / len(smiles_list)
        statistics["unique_valid_ratio"] = float(len(unique_smiles_list)) / len(smiles_list)

        if len(valid_score_list) > 0:
            valid_scores_tsr = torch.FloatTensor(valid_score_list)
            statistics["mae_score"] = (query - valid_scores_tsr).abs().mean()
            statistics["mean_score"] = valid_scores_tsr.mean()
            statistics["std_score"] = valid_scores_tsr.std() if len(result["valid_score_list"]) > 1 else 0.0
            statistics["max_score"] = valid_scores_tsr.max()

            is_success = lambda score: (score > query - success_margin) and (score < query + success_margin)
            num_success = len([score for score in result["unique_score_list"] if is_success(score)])
            statistics["success_ratio"] = float(num_success) / len(smiles_list)

        return result, statistics
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_lstm.parameters(), lr=self.hparams.lr)
        return [optimizer]
