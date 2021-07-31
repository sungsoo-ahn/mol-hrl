from data.graph.util import smiles2graph
import torch.nn as nn

from data.graph.dataset import GraphDataset
from data.graph.transform import mask, fragment_contract, subgraph, fragment
from data.sequence.dataset import SequenceDataset
from module.encoder.graph import GraphEncoder
from module.decoder.sequence import SequenceDecoder


class BaseAutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = GraphEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)

        if hparams.input_graph_fragment_contract:
            self.transform = fragment_contract
        elif hparams.input_graph_mask:
            self.transform = lambda smiles: mask(smiles2graph(smiles))
        elif hparams.input_graph_subgraph:
            self.transform = subgraph
        #elif hparams.input_graph_fragment:
        #    self.transform = fragment
        else:
            self.transform=smiles2graph

    def update_loss(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data = batched_data
        codes, loss, statistics = self.update_encoder_loss(batched_input_data, loss, statistics)
        loss, statistics = self.update_decoder_loss(batched_target_data, codes, loss, statistics)
        return loss, statistics

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        codes = self.encoder(batched_input_data)
        return codes, loss, statistics

    def update_decoder_loss(self, batched_target_data, codes, loss=0.0, statistics=dict()):
        decoder_out = self.decoder(batched_target_data, codes)
        recon_loss, recon_statistics = self.decoder.compute_recon_loss(
            decoder_out, batched_target_data
        )
        loss += recon_loss
        statistics.update(recon_statistics)
        return loss, statistics

    def get_input_dataset(self, split):
        return GraphDataset(self.hparams.data_dir, split, transform=self.transform)

    def get_target_dataset(self, split):
        return SequenceDataset(self.hparams.data_dir, split)

    @staticmethod
    def collate(data_list):
        input_data_list, target_data_list = zip(*data_list)
        batched_input_data = GraphDataset.collate(input_data_list)
        batched_target_data = SequenceDataset.collate(target_data_list)
        return batched_input_data, batched_target_data

    def encode(self, batched_input_data):
        codes = self.encoder(batched_input_data)
        return codes
