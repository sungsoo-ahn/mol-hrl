from argparse import Namespace
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pytorch_lightning as pl

from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_models(hparams)

    def setup_models(self, hparams):
        if hparams.encoder_type == "seq":
            self.encoder = SeqEncoder(hparams)
        elif hparams.encoder_type == "graph":
            self.encoder = GraphEncoder(hparams)

        if hparams.decoder_type == "seq":
            self.decoder = SeqDecoder(hparams)
        
        if hparams.ae_type == "vae":
            self.linear_mu = nn.Linear(hparams.code_dim, hparams.code_dim)
            self.linear_logvar = nn.Linear(hparams.code_dim, hparams.code_dim)
        
        elif hparams.ae_type == "aae":
            self.discriminator = nn.Sequential(
                #nn.Linear(hparams.code_dim, hparams.code_dim),
                #nn.ReLU(),
                nn.Linear(hparams.code_dim, 1),
                nn.Sigmoid(),
            )
        
    @staticmethod
    def add_args(parser):
        # Common
        parser.add_argument("--ae_type", type=str, default="ae")
        parser.add_argument("--encoder_type", type=str, default="seq")
        parser.add_argument("--decoder_type", type=str, default="seq")
        parser.add_argument("--code_dim", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-3)

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

        # VAE specific
        parser.add_argument("--vae_coef", type=float, default=0.1)

        # AAE specific
        parser.add_argument("--aae_coef", type=float, default=0.1)

        # CAE specific
        parser.add_argument("--cae_coef", type=float, default=1.0)

        return parser

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def shared_step(self, batched_data):
        batched_input_data, batched_target_data = batched_data
        encoder_out = self.compute_encoder_out(batched_input_data)
        p, q, codes = self.compute_codes(encoder_out)
        decoder_out = self.compute_decoder_out(batched_target_data, codes)
        loss, statistics = self.compute_loss(
            decoder_out, encoder_out, batched_target_data, p, q, codes
        )
        return loss, statistics

    def compute_encoder_out(self, batched_input_data):
        if isinstance(batched_input_data, list):
            encoder_out = [self.encoder(data) for data in batched_input_data]
        else:
            encoder_out = self.encoder(batched_input_data)

        return encoder_out

    def compute_codes(self, encoder_out):
        if self.hparams.ae_type == "vae":
            mu = self.linear_mu(encoder_out)
            logvar = self.linear_logvar(encoder_out)
            std = torch.exp(logvar / 2)
            p = Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = Normal(mu, std)
            codes = q.rsample()
        elif self.hparams.ae_type == "cae":
            p, q = None, None
            codes = F.normalize(encoder_out[0], p=2, dim=1)
        elif self.hparams.ae_type == "sae":
            p, q = None, None
            codes = F.normalize(encoder_out, p=2, dim=1)
        else:
            p, q = None, None
            codes = encoder_out

        return p, q, codes

    def compute_decoder_out(self, batched_target_data, codes):
        return self.decoder(batched_target_data, codes)

    def compute_loss(self, decoder_out, encoder_out, batched_target_data, p, q, codes):
        statistics = OrderedDict()
        statistics["loss/recon"] = self.decoder.compute_loss(decoder_out, batched_target_data)
        statistics.update(self.decoder.compute_statistics(decoder_out, batched_target_data))
        loss = statistics["loss/recon"]
        if self.hparams.ae_type == "vae":
            vae_loss, vae_statistics = self.compute_vae_reg_loss(p, q, codes)
            loss += self.hparams.vae_coef * vae_loss
            statistics.update(vae_statistics)

        elif self.hparams.ae_type == "aae":
            aae_loss, aae_statistics = self.compute_aae_reg_loss(codes)
            loss += self.hparams.aae_coef * aae_loss
            statistics.update(aae_statistics)

        elif self.hparams.ae_type == "cae":
            cae_loss, cae_statistics = self.compute_cae_reg_loss(encoder_out)
            loss += self.hparams.cae_coef * cae_loss
            statistics.update(cae_statistics)

        return loss, statistics

    def compute_vae_reg_loss(self, p, q, codes):
        loss = (q.log_prob(codes) - p.log_prob(codes)).mean()
        return loss, {"loss/kl": loss}

    def compute_aae_reg_loss(self, codes):
        zn = torch.randn_like(codes)
        zeros = torch.zeros(len(codes), 1, device=self.device)
        ones = torch.ones(len(codes), 1, device=self.device)
        discriminator_out_detach = self.discriminator(codes.detach())
        discriminator_zn_out = self.discriminator(zn)
        discriminator_out = self.discriminator(codes)
        adv_d_loss = F.binary_cross_entropy(
            discriminator_out_detach, zeros
        ) + F.binary_cross_entropy(discriminator_zn_out, ones)
        adv_g_loss = F.binary_cross_entropy(discriminator_out, ones)
        adv_d_acc = (
            0.5 * (discriminator_out_detach < 0.5).float().mean() 
            + 0.5 * (discriminator_zn_out > 0.5).float().mean()
        )

        loss = adv_d_loss + adv_g_loss
        return loss, {"loss/adv_d": adv_d_loss, "loss/adv_g": adv_g_loss, "acc/adv_d": adv_d_acc}

    def compute_cae_reg_loss(self, encoder_out):
        out0, out1 = encoder_out
        codes0 = F.normalize(out0, p=2, dim=1)
        codes1 = F.normalize(out1, p=2, dim=1)
        logits = torch.mm(codes0, codes1.T)
        targets = torch.arange(codes0.size(0)).to(self.device)
        loss = F.cross_entropy(logits, targets)
        acc = (torch.argmax(logits, dim=-1) == targets).float().mean()
        return loss, {"loss/contrastive": loss, "acc/contrastive": acc}
