import torch
import torch.nn.functional as F
from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder
from module.model.base import BaseAEModule

class Seq2SeqCAEModule(BaseAEModule):   
    def setup_models(self, hparams):
        self.decoder = SeqDecoder(hparams)
        self.encoder = SeqEncoder(hparams)

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        SeqEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=1e0)
        return parser

    def shared_step(self, batched_data):
        (batched_input_data0, batched_input_data1), batched_target_data = batched_data
        codes0, encoder_statistics  = self.encoder(batched_input_data0)
        codes1, _ = self.encoder(batched_input_data1)
        codes0 = F.normalize(codes0, p=2, dim=1)
        codes1 = F.normalize(codes1, p=2, dim=1)
        
        contrastive_loss, contrastive_statistics = self.contrastive_loss(codes0, codes1)
        recon_loss, decoder_statistics = self.decoder(batched_target_data, codes0)

        loss = recon_loss + self.hparams.gamma * contrastive_loss

        statistics = {"loss/recon": recon_loss}
        statistics.update(decoder_statistics)
        statistics.update(encoder_statistics)
        statistics.update(contrastive_statistics)

        return loss, statistics

    def contrastive_loss(self, codes0, codes1):
        logits = torch.mm(codes0, codes1.T) 
        targets = torch.arange(codes0.size(0)).to(self.device)
        loss = F.cross_entropy(logits, targets)
        acc = (torch.argmax(logits, dim=-1) == targets).float().mean()
        return loss, {"acc/contrastive": acc}

class Graph2SeqCAEModule(Seq2SeqCAEModule):
    def setup_models(self, hparams):    
        self.decoder = SeqDecoder(hparams)
        self.encoder = GraphEncoder(hparams)

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        GraphEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=1e0)
        return parser