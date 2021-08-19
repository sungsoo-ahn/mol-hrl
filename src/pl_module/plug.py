from argparse import Namespace
from enum import unique

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.factory import load_dataset, load_collate
from data.score.score import BindingScorer, PLogPScorer
from data.score.dataset import PLOGP_MEAN, PLOGP_STD
from data.util import load_tokenizer
from data.smiles.util import canonicalize
from module.factory import load_encoder, load_decoder
from module.vq_layer import VectorQuantizeLayer
from pl_module.autoencoder import AutoEncoderModule

class PlugVariationalAutoEncoder(torch.nn.Module):
    def __init__(self, vq, vq_num_vocabs, code_dim, plug_code_dim, plug_hidden_dim, plug_beta):
        super(PlugVariationalAutoEncoder, self).__init__()
        self.vq = vq
        self.vq_num_vocabs = vq_num_vocabs
        x_dim = code_dim * vq_num_vocabs if vq else code_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, plug_code_dim),
        )   
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(plug_code_dim+1, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, plug_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(plug_hidden_dim, x_dim),
        )

        self.linear_mu = torch.nn.Linear(plug_code_dim, plug_code_dim)
        self.linear_logstd = torch.nn.Linear(plug_code_dim, plug_code_dim)

        self.plug_beta = plug_beta
        self.plug_code_dim = plug_code_dim

    def step(self, x, y):
        loss, statistics = 0.0, dict()
        
        if self.vq:
            out = torch.nn.functional.one_hot(x, self.vq_num_vocabs).float().view(x.size(0), -1)   
        else:
            out = x

        out = self.encoder(out)
        mu = self.linear_mu(out)
        std = (self.linear_logstd(out).exp() + 1e-6)
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        
        z = q.rsample()
        x_hat = self.decoder(torch.cat([z, y.view(-1, 1)], dim=1))
        
        if self.vq:
            x_hat = x_hat.view(-1, self.vq_num_vocabs)
            recon_loss = torch.nn.functional.cross_entropy(x_hat, x.view(-1), reduction="mean")
            statistics["acc/plug"] = (torch.argmax(x_hat, dim=-1) == x.view(-1)).float().mean()
        else:
            recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl_loss = log_qz - log_pz
        kl_loss = kl_loss.mean()
        loss = self.plug_beta * kl_loss + recon_loss

        statistics["loss/plug/recon"] = recon_loss
        statistics["loss/plug/kl"] = kl_loss

        return loss, statistics

    def sample(self, y):
        mu = torch.zeros(y.size(0), self.plug_code_dim, device=y.device)
        std = torch.ones(y.size(0), self.plug_code_dim, device=y.device)
        p = torch.distributions.Normal(mu, std)
        z = p.sample()
        x_hat = self.decoder(torch.cat([z, y], dim=1))
        if self.vq:
            x_hat = torch.argmax(x_hat.view(-1, self.vq_num_vocabs), dim=1).view(y.size(0), -1)
            
        return x_hat


class PlugVariationalAutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(PlugVariationalAutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.tokenizer = load_tokenizer()
        code_dim = hparams.code_dim * hparams.vq_code_dim if hparams.vq else hparams.code_dim
        self.encoder = load_encoder(hparams.encoder_name, code_dim)
        self.decoder = load_decoder(hparams.decoder_name, code_dim)
        self.plug_vae = PlugVariationalAutoEncoder(
            hparams.vq, 
            hparams.vq_num_vocabs, 
            hparams.code_dim, 
            hparams.plug_code_dim, 
            hparams.plug_hidden_dim, 
            hparams.plug_beta
            )
        if self.hparams.vq:
            self.vq_layer = VectorQuantizeLayer(hparams.vq_code_dim, hparams.vq_num_vocabs)

        if hparams.load_checkpoint_path != "":
            pl_ae = AutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint_path)
            self.encoder.load_state_dict(pl_ae.encoder.state_dict())
            self.decoder.load_state_dict(pl_ae.decoder.state_dict())
            if hparams.vq:
                self.vq_layer.load_state_dict(pl_ae.vq_layer.state_dict())

        self.dataset = load_dataset(hparams.dataset_name, hparams.task, hparams.split)
        self.collate = load_collate(hparams.dataset_name)
        
        if hparams.task == "plogp":
            self.scorer = PLogPScorer() 
        elif hparams.task == "binhding":
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
        parser.add_argument("--plug_beta", type=float, default=0.01)
        parser.add_argument("--plug_code_dim", type=int, default=64)
        parser.add_argument("--plug_hidden_dim", type=float, default=1024)
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
            if self.hparams.vq:
                _, codes, _ = self.vq_layer(codes)

        loss, statistics = self.plug_vae.step(codes, batched_cond_data)

        

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
        score_queries = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
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
            codes = self.plug_vae.sample(batched_cond_data)
            if self.hparams.vq:
                codes = self.vq_layer.compute_embedding(codes)
        
            batched_sequence_data = self.decoder.sample(codes, argmax=True, max_len=self.hparams.max_len)
            smiles_list_ = [
                self.tokenizer.decode(data).replace(" ", "") for data in batched_sequence_data.tolist()
                ]
            valid_smiles_list_ = [smi for smi in list(map(canonicalize, smiles_list_)) if smi is not None]

            smiles_list.extend(smiles_list_)
            valid_smiles_list.extend(valid_smiles_list_)
            unique_smiles_list = list(set(valid_smiles_list))

        valid_score_list = self.scorer(valid_smiles_list)
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
        optimizer = torch.optim.Adam(self.plug_vae.parameters(), lr=self.hparams.lr)
        return [optimizer]
