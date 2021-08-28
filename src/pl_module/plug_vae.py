import torch

from module.encoder.gnn import GNNEncoder
from module.decoder.lstm import LSTMDecoder
from module.vq_layer import FlattenedVectorQuantizeLayer
from module.plug.vae import PlugVariationalAutoEncoder
from pl_module.autoencoder import AutoEncoderModule, VectorQuantizedAutoEncoderModule
from pl_module.conddecoder import CondDecoderModule

class PlugVariationalAutoEncoderModule(CondDecoderModule):
    def setup_layers(self, hparams):
        self.encoder = GNNEncoder(
            num_layers=hparams.encoder_num_layers,
            hidden_dim=hparams.encoder_hidden_dim,
            code_dim=hparams.code_dim
        )
        self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=hparams.code_dim
            )
        self.plug_vae = PlugVariationalAutoEncoder(
            num_layers=hparams.plug_num_layers, 
            code_dim=hparams.code_dim, 
            hidden_dim=hparams.plug_hidden_dim, 
            latent_dim=hparams.plug_latent_dim,
            )

        if hparams.load_checkpoint_path != "":
            ae = AutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint_path)
            self.encoder.load_state_dict(ae.encoder.state_dict())
            self.decoder.load_state_dict(ae.decoder.state_dict())

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-3)
    
        # Common - data
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model        
        parser.add_argument("--load_checkpoint_path", type=str, default="")

        # model - code
        parser.add_argument("--code_dim", type=int, default=256)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # model - plug lstm
        parser.add_argument("--plug_num_layers", type=int, default=3)
        parser.add_argument("--plug_hidden_dim", type=int, default=512)
        parser.add_argument("--plug_latent_dim", type=int, default=128)
        parser.add_argument("--plug_beta", type=float, default=1e-1)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=5000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def shared_step(self, batched_data):
        batched_input_data, _, batched_cond_data = batched_data
        self.encoder.eval()
        with torch.no_grad():
            codes = self.encoder(batched_input_data)
        
        out, z, p, q = self.plug_vae(codes, batched_cond_data)
        
        recon_loss = torch.nn.functional.mse_loss(out, codes, reduction="mean")
        kl_loss = (q.log_prob(z) - p.log_prob(z)).mean()
        loss = recon_loss + self.hparams.plug_beta * kl_loss
        
        statistics = dict()        
        statistics["loss/plug_recon"] = recon_loss
        statistics["loss/plug_kl"] = kl_loss

        return loss, statistics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_vae.parameters(), lr=self.hparams.lr)
        return [optimizer]

"""
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
    """
    