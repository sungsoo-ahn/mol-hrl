import torch
import torch.nn.functional as F
from model.ae import AutoEncoderModel
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

class VariationalAutoEncoderModel(AutoEncoderModel):
    def __init__(self, hparams):
        super(VariationalAutoEncoderModel, self).__init__(hparams)
        self.spherical = hparams.spherical
        self.linear_mu = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        if self.spherical:
            self.linear_logvar = torch.nn.Linear(hparams.code_dim, 1)
        else:
            self.linear_logvar = torch.nn.Linear(hparams.code_dim, hparams.code_dim)

    @staticmethod
    def add_args(parser):
        super(VariationalAutoEncoderModel, VariationalAutoEncoderModel).add_args(parser)
        parser.add_argument("--spherical", action="store_true")

    def shared_step(self, batched_data):
        batched_sequence_data, batched_pyg_data, scores = batched_data
        out, _ = self.encoder(batched_pyg_data)
        p, q, codes = self.sample(out)
        logits = self.decoder(batched_sequence_data, codes)
        scores_pred = self.scores_predictor(codes.detach())

        loss_kl = (q.log_prob(codes) - p.log_prob(codes).to(self.device)).mean()
        loss_ce = self.compute_cross_entropy(logits, batched_sequence_data)
        loss_mse = F.mse_loss(scores_pred, scores)
        loss = loss_ce + loss_mse + loss_kl

        acc_elem, acc_sequence = self.compute_accuracy(logits, batched_sequence_data)

        return (
            loss, 
            {
                "loss/ce": loss_ce, 
                "loss/mse": loss_mse, 
                "loss/kl": loss_kl,
                "acc/elem": acc_elem, 
                "acc/sequence": acc_sequence
            },
        )

    def sample(self, out):
        if self.spherical:
            z_mean = self.linear_mu(out)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            z_var = F.softplus(self.linear_logvar(out)) + 1
            q = VonMisesFisher(z_mean, z_var)
            p = HypersphericalUniform(out.size(1) - 1)
        else:
            mu = self.linear_mu(out)
            logvar = self.linear_logvar(out)
            std = torch.exp(logvar / 2)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)

        codes = q.rsample()

        return p, q, codes