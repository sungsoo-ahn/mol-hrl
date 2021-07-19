from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import pytorch_lightning as pl

from hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher

from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder
from data.util import ZipDataset
from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset, RelationalGraphDataset

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class AutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        if hparams.encoder_type == "seq":
            self.encoder = SeqEncoder(hparams)
        elif hparams.encoder_type == "graph":
            self.encoder = GraphEncoder(hparams)

        if hparams.decoder_type == "seq":
            self.decoder = SeqDecoder(hparams)
    
    @staticmethod
    def add_args(parser):
        return parser

    def compute_loss(self, batched_data):
        loss, statistics = 0.0, dict()
        codes, loss, statistics = self.update_encoder_loss(batched_data, loss, statistics)
        loss, statistics = self.update_decoder_loss(batched_data, codes, loss, statistics)
        return loss, statistics
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        codes = self.encoder(batched_input_data)
        return codes, loss, statistics
    
    def update_decoder_loss(self, batched_data, codes, loss, statistics):
        _, batched_target_data = batched_data
        decoder_out = self.decoder(batched_target_data, codes)
        recon_loss, recon_statistics = self.decoder.compute_recon_loss(
            decoder_out, batched_target_data
            )
        
        loss += recon_loss
        statistics.update(recon_statistics)

        return loss, statistics
        
    def encode(self, batched_input_data):
        codes = self.encoder(batched_input_data)
        return codes

    def project(self, encoder_out, batched_target_data=None):
        return encoder_out

class SphericalAutoEncoder(AutoEncoder):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--sae_norm_loss_coef", type=float, default=0.01)
        parser.add_argument("--sae_uniform_loss_coef", type=float, default=0.0)
        parser.add_argument("--sae_vmf_scale", type=float, default=0.0)
        return parser

    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, batched_target_data = batched_data
        encoder_out = self.encoder(batched_input_data)
        codes = self.project(encoder_out, batched_target_data)
        encoder_norm_loss = torch.norm(encoder_out, p=2, dim=-1).mean()
        
        loss += self.hparams.sae_norm_loss_coef * encoder_norm_loss 
        statistics["loss/encoder/norm"] = encoder_norm_loss

        if self.hparams.sae_uniform_loss_coef > 0.0:
            dists = (codes @ codes.T).flatten()
            encoder_uniform_loss = dists.mul(-1).exp().mean().log()

            loss += self.hparams.sae_uniform_loss_coef * encoder_uniform_loss
            statistics["loss/encoder/uniform"] = encoder_uniform_loss

        return codes, loss, statistics

    def project(self, encoder_out, batched_target_data=None):
        codes = F.normalize(encoder_out, p=2, dim=1)

        if self.training and self.hparams.sae_vmf_scale > 0.0:
            scale = torch.full((codes.size(0), 1), self.hparams.sae_vmf_scale, device=codes.device)
            m = VonMisesFisher(codes, scale)
            codes = m.rsample()

        return codes

    def encode(self, batched_input_data):
        encoder_out = self.encoder(batched_input_data)
        codes = F.normalize(encoder_out, p=2, dim=-1)
        
        return codes

class RelationalAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        if hparams.encoder_type == "seq":
            self.encoder = SeqEncoder(hparams)
        elif hparams.encoder_type == "graph":
            self.encoder = GraphEncoder(hparams)

        if hparams.decoder_type == "seq":
            self.decoder = SeqDecoder(hparams)

        self.projector = nn.Linear(hparams.code_dim, hparams.code_dim)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--norel", action="store_true")
        return parser

    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        batched_input_data0, batched_input_data1, action_feats = batched_input_data 
        
        if self.hparams.norel:
            codes = codes0 = self.encoder(batched_input_data0)
            codes1 = self.encoder(batched_input_data1)
        else:
            codes, codes0 = self.encoder.forward_cond(batched_input_data0, action_feats)
            codes1 = self.encoder(batched_input_data1)

        out0 = F.normalize(self.projector(codes0), p=2, dim=1)
        out1 = F.normalize(self.projector(codes1), p=2, dim=1)
        logits = out0 @ out1.T
        labels = torch.arange(out0.size(0), device = logits.device)
        relational_loss = F.cross_entropy(logits, labels)

        loss += relational_loss
        statistics["loss/relational"] = relational_loss

        return codes, loss, statistics
        
    def encode(self, batched_input_data):
        codes = self.encoder(batched_input_data)
        return codes

class SphericalRelationalAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = GraphEncoder(hparams)
        self.decoder = SeqDecoder(hparams)

    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        batched_input_data0, batched_input_data1, action_feats = batched_input_data 
        
        codes, codes0 = self.encoder.forward_cond(batched_input_data0, action_feats)
        codes1 = self.encoder(batched_input_data1)

        codes = F.normalize(codes)

        out0 = F.normalize(codes0, p=2, dim=1)
        out1 = F.normalize(codes1, p=2, dim=1)
        logits = out0 @ out1.T
        labels = torch.arange(out0.size(0), device = logits.device)
        relational_loss = F.cross_entropy(logits, labels)

        loss += relational_loss
        statistics["loss/relational"] = relational_loss

        return codes, loss, statistics
        
    def encode(self, batched_input_data):
        out = self.encoder(batched_input_data)
        codes = F.normalize(out, p=2, dim=-1)
        return codes

    def project(self, encoder_out, batched_target_data=None):
        return F.normalize(encoder_out, p=2 ,dim=-1)

class DGIAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = GraphEncoder(hparams)
        self.decoder = SeqDecoder(hparams)
        self.weight = nn.Parameter(torch.Tensor(hparams.code_dim, hparams.code_dim))
        torch_geometric.nn.inits.uniform(self.weight.size(0), self.weight)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dgi_loss_coef", type=float, default=0.1)
        return parser

    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        
        codes, noderep = self.encoder.forward_reps(batched_input_data)
        
        graphrep = torch.sigmoid(codes)
        positive_expanded_summary_emb = graphrep[batched_input_data.batch]
        shifted_summary_emb = graphrep[cycle_index(len(graphrep), 1)]
        negative_expanded_summary_emb = shifted_summary_emb[batched_input_data.batch]

        positive_score = (positive_expanded_summary_emb @ self.weight) @ noderep.T
        negative_score = (negative_expanded_summary_emb @ self.weight) @ noderep.T
        

        dgi_loss = (
            F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score)) 
            + F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
        )
        
        loss += self.hparams.dgi_loss_coef * dgi_loss
        statistics["loss/dgi"] = dgi_loss

        return codes, loss, statistics
        
    def encode(self, batched_input_data):
        out = self.encoder(batched_input_data)
        codes = F.normalize(out, p=2, dim=-1)
        return codes

    def project(self, encoder_out, batched_target_data=None):
        return F.normalize(encoder_out, p=2 ,dim=-1)

class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
    
        if hparams.ae_type in ["srae", "rae"]:
            self.train_input_dataset = RelationalGraphDataset(hparams.data_dir, "train")
            self.val_input_dataset = RelationalGraphDataset(hparams.data_dir, "train_labeled")
            
        elif hparams.encoder_type == "seq":
            self.train_input_dataset = SequenceDataset(hparams.data_dir, "train")
            self.val_input_dataset = SequenceDataset(hparams.data_dir, "train_labeled")
            
        elif hparams.encoder_type == "graph":
            self.train_input_dataset = GraphDataset(hparams.data_dir, "train")
            self.val_input_dataset = GraphDataset(hparams.data_dir, "train_labeled")

        if hparams.decoder_type == "seq":
            self.train_target_dataset = SequenceDataset(
                hparams.data_dir, "train", use_random_smiles=hparams.use_random_smiles
                )
            self.val_target_dataset = SequenceDataset(
                hparams.data_dir, "train_labeled", use_random_smiles=hparams.use_random_smiles
                )
            hparams.num_vocabs = len(self.train_target_dataset.vocabulary)

        self.train_dataset = ZipDataset(self.train_input_dataset, self.train_target_dataset)
        self.val_dataset = ZipDataset(self.val_input_dataset, self.val_target_dataset)
    
        if hparams.ae_type == "ae":
            self.ae = AutoEncoder(hparams)
        elif hparams.ae_type == "sae":
            self.ae = SphericalAutoEncoder(hparams)
        elif hparams.ae_type == "rae":
            self.ae = RelationalAutoEncoder(hparams)
        elif hparams.ae_type == "srae":
            self.ae = SphericalRelationalAutoEncoder(hparams)
        elif hparams.ae_type == "dgi_ae":
            self.ae = DGIAutoEncoder(hparams)

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--ae_type", type=str, default="ae")
        parser.add_argument("--encoder_type", type=str, default="seq")
        parser.add_argument("--decoder_type", type=str, default="seq")
        parser.add_argument("--code_dim", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-3)

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        # GraphEncoder specific
        parser.add_argument("--graph_encoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--graph_encoder_num_layers", type=int, default=5)

        # SeqEncoder specific
        parser.add_argument("--seq_encoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--seq_encoder_num_layers", type=int, default=3)

        # SecDecoder specific
        parser.add_argument("--seq_decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--seq_decoder_num_layers", type=int, default=3)
        parser.add_argument("--seq_decoder_max_length", type=int, default=81)

        parser.add_argument("--use_random_smiles", action="store_true")

        # AutoEncoder specific
        AutoEncoder.add_args(parser)

        # SAE specific
        SphericalAutoEncoder.add_args(parser)

        #
        RelationalAutoEncoder.add_args(parser)

        #
        DGIAutoEncoder.add_args(parser)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.ae.compute_loss(batched_data)
        self.log("train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.ae.compute_loss(batched_data)
        self.log("validation/loss/total", loss, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]