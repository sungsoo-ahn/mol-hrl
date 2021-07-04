import torch

from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder
from module.model.base import BaseAEModule

class Seq2SeqVAEModule(BaseAEModule):   
    def setup_models(self, hparams):
        self.decoder = SeqDecoder(hparams)
        self.encoder = SeqEncoder(hparams)
        self.linear_mu = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        self.linear_logvar = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
    
    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        SeqEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta", type=float, default=1e-1)
        return parser

    def shared_step(self, batched_data):
        batched_input_data, batched_target_data = batched_data
        out, encoder_statistics  = self.encoder(batched_input_data)
        kl_loss, codes = self.sample(out)
        recon_loss, decoder_statistics = self.decoder(batched_target_data, codes)

        loss = recon_loss + self.hparams.beta * kl_loss

        statistics = {"loss/kl": kl_loss, "loss/recon": recon_loss}
        statistics.update(decoder_statistics)
        statistics.update(encoder_statistics)

        return loss, statistics

    def sample(self, out):
        mu = self.linear_mu(out)
        logvar = self.linear_logvar(out)
        std = torch.exp(logvar / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        codes = q.rsample()

        kl_loss = (q.log_prob(codes) - p.log_prob(codes)).mean()
        return kl_loss, codes


class Graph2SeqVAEModule(Seq2SeqVAEModule):
    def setup_models(self, hparams):    
        self.decoder = SeqDecoder(hparams)
        self.encoder = GraphEncoder(hparams)
        self.linear_mu = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        self.linear_logvar = torch.nn.Linear(hparams.code_dim, hparams.code_dim)

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        GraphEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta", type=float, default=1e-1)
        return parser