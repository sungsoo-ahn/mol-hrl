import torch
import torch.nn as nn
import torch.nn.functional as F

from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder
from module.model.base import BaseAEModule

class Seq2SeqAAEModule(BaseAEModule):   
    def setup_models(self, hparams):
        self.decoder = SeqDecoder(hparams)
        self.encoder = SeqEncoder(hparams)
        self.discriminator = nn.Sequential(
            nn.Linear(hparams.code_dim, hparams.code_dim),
            nn.ReLU(),
            nn.Linear(hparams.code_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        SeqEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta", type=float, default=1e-1)
        return parser

    def shared_step(self, batched_data):
        batched_input_data, batched_target_data = batched_data
        codes, encoder_statistics  = self.encoder(batched_input_data)
        loss_recon, decoder_statistics = self.decoder(batched_target_data, codes)
        loss_adv = self.loss_adv(codes)

        loss = loss_recon + self.hparams.beta * loss_adv

        statistics = {"loss/recon": loss, "loss/adv": loss_adv}
        statistics.update(decoder_statistics)
        statistics.update(encoder_statistics)

        return loss, statistics

    def loss_adv(self, codes):
        zn = torch.randn_like(codes)
        zeros = torch.zeros(len(codes), 1, device=self.device)
        ones = torch.ones(len(codes), 1, device=self.device)
        loss_d = (
            F.binary_cross_entropy(self.discriminator(codes.detach()), zeros) 
            + F.binary_cross_entropy(self.discriminator(zn), ones)
        )
        loss_g = F.binary_cross_entropy(self.discriminator(codes), ones)
        
        return loss_d + loss_g

class Graph2SeqAAEModule(Seq2SeqAAEModule):
    def setup_models(self, hparams):    
        self.decoder = SeqDecoder(hparams)
        self.encoder = GraphEncoder(hparams)
        self.discriminator = nn.Sequential(
            nn.Linear(hparams.code_dim, hparams.code_dim),
            nn.ReLU(),
            nn.Linear(hparams.code_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        GraphEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta", type=float, default=1e-1)
        return parser