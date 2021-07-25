import torch
import torch.nn as nn

from module.autoencoder.base import BaseAutoEncoder


class VariationalAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(VariationalAutoEncoder, self).__init__(hparams)
        self.hparams = hparams
        self.linear_mu = nn.Linear(hparams.code_dim, hparams.code_dim)
        self.linear_logvar = nn.Linear(hparams.code_dim, hparams.code_dim)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--variational_kl_coef", type=float, default=0.1)

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        out = self.encoder(batched_input_data)

        mu = self.linear_mu(out)
        log_var = self.linear_logvar(out)

        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        codes = q.rsample()

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl_loss = (log_qz - log_pz).mean()
        
        loss += self.hparams.variational_kl_coef * kl_loss
        statistics["loss/kl"] = kl_loss

        return codes, loss, statistics

    def encode(self, batched_input_data):
        out = self.encoder(batched_input_data)
        codes = self.linear_mu(out)
        return codes
