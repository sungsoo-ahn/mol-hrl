import torch

from module.encoder.gnn import GNNEncoder
from module.decoder.lstm import LSTMDecoder
from module.vq_layer import FlattenedVectorQuantizeLayer
from module.plug.vae import PlugVariationalAutoEncoder, PlugDiscreteVariationalAutoEncoder
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
        parser.add_argument("--lr", type=float, default=1e-4)
    
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
        parser.add_argument("--plug_hidden_dim", type=int, default=1024)
        parser.add_argument("--plug_latent_dim", type=int, default=256)
        parser.add_argument("--plug_beta", type=float, default=1e0)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=1000)
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
        statistics["loss/total"] = loss
        statistics["loss/plug_recon"] = recon_loss
        statistics["loss/plug_kl"] = kl_loss

        return loss, statistics

    def decode(self, query, num_samples, max_len):
        batched_cond_data = torch.full((num_samples, 1), query, device=self.device)
        batched_cond_data = (batched_cond_data - self.score_mean) / self.score_std
        codes = self.plug_vae.decode(batched_cond_data)
        batched_sequence_data = self.decoder.decode(codes, argmax=True, max_len=max_len)
        return batched_sequence_data
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_vae.parameters(), lr=self.hparams.lr)
        return [optimizer]

class PlugDiscreteVariationalAutoEncoderModule(CondDecoderModule):
    def setup_layers(self, hparams):
        code_dim = hparams.vq_code_dim * hparams.vq_codebook_dim
        self.encoder = GNNEncoder(
            num_layers=hparams.encoder_num_layers,
            hidden_dim=hparams.encoder_hidden_dim,
            code_dim=code_dim
        )
        self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=code_dim
            )
        self.vq_layer = FlattenedVectorQuantizeLayer(hparams.vq_codebook_dim, hparams.vq_num_vocabs)
        self.plug_vqvae = PlugDiscreteVariationalAutoEncoder(
            num_layers=hparams.plug_num_layers, 
            vq_code_dim=hparams.vq_code_dim, 
            vq_num_vocabs=hparams.vq_num_vocabs,
            hidden_dim=hparams.plug_hidden_dim, 
            latent_dim=hparams.plug_latent_dim,
            )

        if hparams.load_checkpoint_path != "":
            vqae = VectorQuantizedAutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint_path)
            self.encoder.load_state_dict(vqae.encoder.state_dict())
            self.decoder.load_state_dict(vqae.decoder.state_dict())
            self.vq_layer.load_state_dict(vqae.vq_layer.state_dict())


    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-4)
    
        # Common - data
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model        
        parser.add_argument("--load_checkpoint_path", type=str, default="")

        # model - code
        parser.add_argument("--vq_code_dim", type=int, default=64)
        parser.add_argument("--vq_codebook_dim", type=int, default=128)
        parser.add_argument("--vq_num_vocabs", type=int, default=64)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # model - plug lstm
        parser.add_argument("--plug_num_layers", type=int, default=2)
        parser.add_argument("--plug_hidden_dim", type=int, default=64)
        parser.add_argument("--plug_latent_dim", type=int, default=32)
        parser.add_argument("--plug_beta", type=float, default=1e0)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=1000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def shared_step(self, batched_data):
        batched_input_data, _, batched_cond_data = batched_data
        self.encoder.eval()
        with torch.no_grad():
            codes = self.encoder(batched_input_data)
            _, code_idxs, _ = self.vq_layer(codes)
        
        x_hat, z, p, q = self.plug_vqvae(code_idxs, batched_cond_data)

        recon_loss = torch.nn.functional.cross_entropy(x_hat, code_idxs.view(-1), reduction="mean")
        kl_loss = (q.log_prob(z) - p.log_prob(z)).mean()
        loss = recon_loss + self.hparams.plug_beta * kl_loss
        
        statistics = dict()        
        statistics["loss/total"] = recon_loss
        statistics["loss/plug_recon"] = recon_loss
        statistics["loss/plug_kl"] = kl_loss
        statistics["acc/plug"] = (torch.argmax(x_hat, dim=-1) == code_idxs.view(-1)).float().mean()

        return loss, statistics

    def decode(self, query, num_samples, max_len):
        batched_cond_data = torch.full((num_samples, 1), query, device=self.device)
        batched_cond_data = (batched_cond_data - self.score_mean) / self.score_std
        code_idxs = self.plug_vqvae.decode(batched_cond_data)
        codes = self.vq_layer.compute_embedding(code_idxs)
        batched_sequence_data = self.decoder.decode(codes, argmax=True, max_len=max_len)
        return batched_sequence_data
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_vqvae.parameters(), lr=self.hparams.lr)
        return [optimizer]
