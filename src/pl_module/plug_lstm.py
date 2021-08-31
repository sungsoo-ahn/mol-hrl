import torch
import torch.nn as nn

from module.encoder.gnn import GNNEncoder
from module.decoder.lstm import LSTMDecoder
from module.plug.lstm import PlugLSTM
from module.vq_layer import FlattenedVectorQuantizeLayer
from pl_module.conddecoder import CondDecoderModule
from pl_module.autoencoder import VectorQuantizedAutoEncoderModule

def compute_sequence_cross_entropy(logits, batched_sequence_data):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )

    return loss
   
class PlugLSTMModule(CondDecoderModule):
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
        self.plug_lstm = PlugLSTM(
            hparams.plug_num_layers, hparams.plug_hidden_dim, hparams.vq_code_dim, hparams.vq_num_vocabs
            )

        if hparams.load_checkpoint_path != "":
            vqae = VectorQuantizedAutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint_path)
            self.encoder.load_state_dict(vqae.encoder.state_dict())
            self.decoder.load_state_dict(vqae.decoder.state_dict())
            self.vq_layer.load_state_dict(vqae.vq_layer.state_dict())

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-3)
    
        # Common - data
        parser.add_argument("--task", type=str, default="plogp")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=256)
        
        # model        
        parser.add_argument("--load_checkpoint_path", type=str, default="")

        # model - code
        parser.add_argument("--vq_code_dim", type=int, default=256)
        parser.add_argument("--vq_codebook_dim", type=int, default=256)
        parser.add_argument("--vq_num_vocabs", type=int, default=64)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # model - plug lstm
        parser.add_argument("--plug_num_layers", type=int, default=2)
        parser.add_argument("--plug_hidden_dim", type=int, default=512)
        
        # sampling
        parser.add_argument("--num_queries", type=int, default=1000)
        parser.add_argument("--query_batch_size", type=int, default=250)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data, batched_cond_data = batched_data
        self.encoder.eval()
        with torch.no_grad():
            codes = self.encoder(batched_input_data)
            codes, code_idxs, _ = self.vq_layer(codes)

        logits = self.plug_lstm(code_idxs, batched_cond_data)

        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), code_idxs.view(-1))

        preds = torch.argmax(logits, dim=-1)
        correct = (preds == code_idxs)
        elem_acc = correct.float().mean()
        sequence_acc = correct.view(code_idxs.size(0), -1).all(dim=1).float().mean()

        statistics = {
            "loss/plug_recon": loss,
            "acc/plug_elem": elem_acc,
            "acc/plug_seq": sequence_acc
        }

        return loss, statistics

    def decode(self, query, num_samples, max_len):
        batched_cond_data = torch.full((num_samples, 1), query, device=self.device)
        batched_cond_data = (batched_cond_data - self.score_mean) / self.score_std
        codes = self.plug_lstm.decode(batched_cond_data)
        codes = self.vq_layer.compute_embedding(codes)    
        batched_sequence_data = self.decoder.decode(codes, argmax=True, max_len=max_len)
        return batched_sequence_data
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.plug_lstm.parameters(), lr=self.hparams.lr)
        return [optimizer]
