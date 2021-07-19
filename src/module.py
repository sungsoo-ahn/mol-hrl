from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import pytorch_lightning as pl

from net.seq import SeqDecoder
from net.graph import GraphEncoder
from data.util import ZipDataset
from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset, RelationalGraphDataset, ContrastiveGraphDataset


class AutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = GraphEncoder(hparams)
        self.decoder = SeqDecoder(hparams)
    
    def compute_loss(self, batched_data):
        loss, statistics = 0.0, dict()
        codes, loss, statistics = self.update_encoder_loss(batched_data, loss, statistics)
        loss, statistics = self.update_decoder_loss(batched_data, codes, loss, statistics)
        return loss, statistics
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        codes = self.encoder(batched_input_data)
        loss, statistics = self.update_code_norm_loss(codes, loss, statistics)        
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
    
    def update_code_norm_loss(self, codes, loss, statistics):
        code_norm_loss = torch.norm(codes, p=2, dim=1).mean()
        loss += self.hparams.code_norm_loss_coef * code_norm_loss
        statistics["loss/code_norm"] = code_norm_loss.item()
        return loss, statistics

    def get_input_dataset(self, split):
        return GraphDataset(
            self.hparams.data_dir, 
            split, 
            smiles_transform_type=self.hparams.input_smiles_transform_type,
            graph_transform_type=self.hparams.input_graph_transform_type
            )
    
    def get_target_dataset(self, split):
        return SequenceDataset(
            self.hparams.data_dir, 
            split,       
            smiles_transform_type=self.hparams.target_smiles_transform_type,
            seq_transform_type=self.hparams.target_seq_transform_type
            )

class SupervisedAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(SupervisedAutoEncoder, self).__init__(hparams)
        self.regressor = nn.Linear(hparams.code_dim, 1)
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        (batched_input_data, regression_target), _ = batched_data
        codes = self.encoder(batched_input_data)
        loss, statistics = self.update_code_norm_loss(codes, loss, statistics)        
        loss, statistics = self.update_regression_loss(codes, regression_target, loss, statistics)
        return codes, loss, statistics
    
    def update_regression_loss(self, codes, regression_target, loss, statistics):
        regression_pred = self.regressor(codes)
        regression_loss = F.mse_loss(regression_pred, regression_target)
        loss += regression_loss
        statistics["loss/regression"] = regression_loss.item()
        return loss, statistics    
    
class ContrastiveAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(ContrastiveAutoEncoder, self).__init__(hparams)
        self.projector = nn.Linear(hparams.code_dim, hparams.code_dim)
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        (batched_input_data0, batched_input_data1), _ = batched_data
        codes = codes0 = self.encoder(batched_input_data0)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics
    
    def update_contrastive_loss(self, codes0, codes1, loss, statistics):
        out0 = F.normalize(self.projector(codes0), p=2, dim=1)
        out1 = F.normalize(self.projector(codes1), p=2, dim=1)
        logits = out0 @ out1.T
        labels = torch.arange(out0.size(0), device = logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        contrastive_acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        loss += contrastive_loss
        statistics["loss/contrastive"] = contrastive_loss
        statistics["acc/contrastive"] = contrastive_acc
        return loss, statistics

    def get_input_dataset(self, split):
        return ContrastiveGraphDataset(
            self.hparams.data_dir, 
            split,
            smiles_transform_type=self.hparams.input_smiles_transform_type,
            graph_transform_type=self.hparams.input_graph_transform_type
            )

class RelationalAutoEncoder(ContrastiveAutoEncoder):
    def __init__(self, hparams):
        super(RelationalAutoEncoder, self).__init__(hparams)
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        (batched_input_data0, batched_input_data1, action_feats), _ = batched_data
        codes, codes0 = self.encoder.forward_cond(batched_input_data0, action_feats)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics

    def get_input_dataset(self, split):
        return RelationalGraphDataset(self.hparams.data_dir, split)
    
class DGIAutoEncoder(AutoEncoder):
    def __init__(self, hparams):
        super(DGIAutoEncoder, self).__init__(hparams)
        self.dgi_mat = nn.Parameter(torch.Tensor(hparams.code_dim, hparams.code_dim))
        torch_geometric.nn.inits.uniform(self.dgi_mat.size(0), self.dgi_mat)

    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, _ = batched_data
        
        codes, noderep = self.encoder.forward_reps(batched_input_data)
        graphrep = torch.sigmoid(codes)

        loss, statistics = self.update_code_norm_loss(codes, loss, statistics)
        loss, statistics = self.update_dgi_loss(
            noderep, graphrep, batched_input_data.batch, loss, statistics
            )
        return codes, loss, statistics

    def update_dgi_loss(self, noderep, graphrep, batch, loss, statistics):
        positive_expanded_summary_emb = graphrep[batch]
        
        cycle_index = torch.arange(len(graphrep)) + 1
        cycle_index[-1] = 0
        shifted_summary_emb = graphrep[cycle_index]
        negative_expanded_summary_emb = shifted_summary_emb[batch]

        positive_score = (positive_expanded_summary_emb @ self.dgi_mat) @ noderep.T
        negative_score = (negative_expanded_summary_emb @ self.dgi_mat) @ noderep.T

        dgi_loss = (
            F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score)) 
            + F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
        )
        
        loss += self.hparams.dgi_loss_coef * dgi_loss
        statistics["loss/dgi"] = dgi_loss

        return loss, statistics

        
class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        
        if hparams.ae_type == "ae":
            self.ae = AutoEncoder(hparams)
        elif hparams.ae_type == "vae":
            self.ae = VariationalAutoEncoder(hparams)
        elif hparams.ae_type == "con_ae":
            self.ae = ContrastiveAutoEncoder(hparams)
        elif hparams.ae_type == "rel_ae":
            self.ae = RelationalAutoEncoder(hparams)
        elif hparams.ae_type == "dgi_ae":
            self.ae = DGIAutoEncoder(hparams)
        
        self.train_input_dataset = self.ae.get_input_dataset("train")
        self.train_target_dataset = self.ae.get_target_dataset("train")
        self.val_input_dataset = self.ae.get_input_dataset("val")
        self.val_target_dataset = self.ae.get_target_dataset("val")
        self.train_dataset = ZipDataset(self.train_input_dataset, self.train_target_dataset)
        self.val_dataset = ZipDataset(self.val_input_dataset, self.val_target_dataset)
    
    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--ae_type", type=str, default="ae")
        parser.add_argument("--code_dim", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)

        # Common - data
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small/")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        
        parser.add_argument("--input_smiles_transform_type", type=str, default="none")
        parser.add_argument("--input_seq_transform_type", type=str, default="none")
        parser.add_argument("--input_graph_transform_type", type=str, default="none")
        
        parser.add_argument("--target_smiles_transform_type", type=str, default="none")
        parser.add_argument("--target_seq_transform_type", type=str, default="none")
        parser.add_argument("--target_graph_transform_type", type=str, default="none")
        
        # GraphEncoder specific
        parser.add_argument("--graph_encoder_hidden_dim", type=int, default=256)
        parser.add_argument("--graph_encoder_num_layers", type=int, default=5)

        # SeqEncoder specific
        parser.add_argument("--seq_encoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--seq_encoder_num_layers", type=int, default=3)

        # SecDecoder specific
        parser.add_argument("--seq_decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--seq_decoder_num_layers", type=int, default=3)
        parser.add_argument("--seq_decoder_max_length", type=int, default=81)

        # AutoEncoder specific
        parser.add_argument("--code_norm_loss_coef", type=float, default=0.0)
        parser.add_argument("--dgi_loss_coef", type=float, default=0.1)

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