from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from module.encoder.graph import GraphEncoder
from module.decoder.sequence import SequenceDecoder
from data.graph.dataset import GraphDataset
from data.score.dataset import ScoreDataset
from data.score.factory import get_scoring_func
from data.util import ZipDataset


class ConditionalGaussianMixture(torch.nn.Module):
    def __init__(self, hparams):
        super(ConditionalGaussianMixture, self).__init__()
        self.num_mixtures = hparams.num_mixtures
        self.code_dim = hparams.code_dim
        hidden_dim = hparams.num_mixtures * hparams.code_dim
        self.linear_mu = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.linear_logstd = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.mix_params = torch.nn.Parameter(torch.ones(hparams.num_mixtures))

    def forward(self, x):
        dist = self.get_distribution(x)
        return dist.mean

    def log_prob(self, x, code):
        dist = self.get_distribution(x)
        log_probs = dist.log_prob(code)
        return log_probs

    def get_distribution(self, x):
        mus = self.linear_mu(x).view(-1, self.num_mixtures, self.code_dim)
        stds = self.linear_logstd(x).view(-1, self.num_mixtures, self.code_dim).exp() + 1e-3
        mix_params = self.mix_params.view(1, -1).repeat(x.size(0), 1)

        mix = torch.distributions.Categorical(mix_params)
        comp = torch.distributions.Independent(torch.distributions.Normal(mus, stds), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        return gmm

    
def collate(data_list):
    cond_data_list, input_data_list = zip(*data_list)
    batched_cond_data = ScoreDataset.collate(cond_data_list)
    batched_input_data = GraphDataset.collate(input_data_list)
    return batched_cond_data, batched_input_data


class CondDecoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CondDecoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.encoder = GraphEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)
        self.cond_embedding = ConditionalGaussianMixture(hparams)

        if hparams.load_checkpoint_path != "":
            state_dict = torch.load(hparams.load_checkpoint_path)
            if "encoder" in state_dict:
                self.encoder.load_state_dict(state_dict["encoder"])
            elif "cond_embedding" in state_dict:
                self.cond_embedding.load_state_dict(state_dict["cond_embedding"])

        self.train_cond_dataset = ScoreDataset(hparams.data_dir, hparams.score_func_name, hparams.train_split)
        self.train_input_dataset = GraphDataset(hparams.data_dir, hparams.train_split)
        self.train_dataset = ZipDataset(self.train_cond_dataset, self.train_input_dataset)
        
        _, self.score_func, self.corrupt_score = get_scoring_func(hparams.score_func_name)

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--cond_embedding_mlp", action="store_true")

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--load_checkpoint_path", type=str, default="")
        parser.add_argument("--train_split", type=str, default="train_256")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_queries", type=int, default=10000)
        parser.add_argument("--query_batch_size", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--score_func_name", type=str, default="penalized_logp")

        #
        parser.add_argument("--code_dim", type=int, default=256)

        # GraphEncoder specific
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)
        parser.add_argument("--encoder_num_layers", type=int, default=5)

        # SequentialDecoder specific
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_max_length", type=int, default=120)

        # Cond embedding specific
        parser.add_argument("--num_mixtures", type=int, default=10)


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
        self.encoder.eval()

        loss, statistics = 0.0, dict()
        batched_cond_data, batched_input_data = batched_data

        codes = self.encoder(batched_input_data)
        log_probs = self.cond_embedding.log_prob(batched_cond_data, codes)
        loss -= (log_probs.mean() / self.hparams.code_dim)

        statistics["train/log_probs"] = log_probs.mean()
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("train/loss/total", loss, on_step=False, on_epoch=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss

    def training_epoch_end(self, processed_epoch_output):
        if (self.current_epoch + 1) % 100 == 0:
            self.cond_embedding.eval()
            self.decoder.eval()
        
            if self.hparams.score_func_name == "penalized_logp":
                score_queries = [4.0, 5.0, 6.0, 7.0]
            elif self.hparams.score_func_name == "logp":
                score_queries = [4.0, 5.0, 6.0, 7.0]
            elif self.hparams.score_func_name == "molwt":
                score_queries = [500.0, 600.0, 700.0, 800.0]
            elif self.hparams.score_func_name == "qed":
                score_queries = [0.7, 0.8, 0.9, 1.0]

            for query in score_queries:
                smiles_list = []
                for _ in range(self.hparams.num_queries // self.hparams.query_batch_size):
                    query_tsr = torch.full((self.hparams.query_batch_size, 1), query, device=self.device)
                    batched_cond_data = self.train_cond_dataset.normalize(query_tsr)
                    with torch.no_grad():
                        codes = self.cond_embedding(batched_cond_data)

                    smiles_list_ = self.decoder.sample_smiles(codes, argmax=False)
                    smiles_list.extend(smiles_list_)
                
                score_list = self.score_func(smiles_list)

                valid_idxs = [idx for idx, score in enumerate(score_list) if score > self.corrupt_score]
                valid_smiles_list = [smiles_list[idx] for idx in valid_idxs]
                valid_scores = torch.FloatTensor([score_list[idx] for idx in valid_idxs])

                valid_ratio = valid_scores.size(0) / len(score_list)
                self.log(f"query{query:.2f}/valid_ratio", valid_ratio, on_step=False, logger=True)

                if valid_ratio > 0.0:
                    mae = (query - valid_scores).abs().mean()
                    self.log(f"query{query:.2f}/mae", mae, on_step=False, logger=True)

                    unique_ratio = float(len(set(valid_smiles_list))) / len(smiles_list)
                    self.log(f"query{query:.2f}/unique_ratio", unique_ratio, on_step=False, logger=True)
        
    def configure_optimizers(self):
        params = list(self.cond_embedding.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return [optimizer]
