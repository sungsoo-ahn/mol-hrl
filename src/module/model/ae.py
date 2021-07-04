from net.seq import SeqEncoder, SeqDecoder
from net.graph import GraphEncoder
from module.model.base import BaseAEModule

class Seq2SeqAEModule(BaseAEModule):   
    def setup_models(self, hparams):
        self.decoder = SeqDecoder(hparams)
        self.encoder = SeqEncoder(hparams)

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        SeqEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def shared_step(self, batched_data):
        batched_input_data, batched_target_data = batched_data
        codes, encoder_statistics  = self.encoder(batched_input_data)
        loss, decoder_statistics = self.decoder(batched_target_data, codes)

        statistics = {"loss/recon": loss}
        statistics.update(decoder_statistics)
        statistics.update(encoder_statistics)

        return loss, statistics

class Graph2SeqAEModule(Seq2SeqAEModule):
    def setup_models(self, hparams):    
        self.decoder = SeqDecoder(hparams)
        self.encoder = GraphEncoder(hparams)

    @staticmethod
    def add_args(parser):
        SeqDecoder.add_args(parser)
        GraphEncoder.add_args(parser)
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser