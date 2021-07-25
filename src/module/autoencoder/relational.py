import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from data.graph.dataset import GraphDataset
from data.graph.util import smiles2graph
from data.graph.transform import mutate
from data.smiles.util import load_smiles_list

from module.autoencoder.base import BaseAutoEncoder
from module.autoencoder.contrastive import ContrastiveAutoEncoder
from module.encoder.graph import GraphEncoder 
from module.decoder.sequence import SequenceDecoder

class RelationalEncoder(GraphEncoder):
    def __init__(self, hparams):
        super(RelationalEncoder, self).__init__(hparams)
        self.cond_embedding0 = torch.nn.Embedding(5, hparams.graph_encoder_hidden_dim)
        self.cond_embedding1 = torch.nn.Embedding(120, hparams.graph_encoder_hidden_dim)
        self.cond_embedding2 = torch.nn.Embedding(3, hparams.graph_encoder_hidden_dim)
        self.cond_embedding3 = torch.nn.Embedding(6, hparams.graph_encoder_hidden_dim)
        self.cond_embedding4 = torch.nn.Embedding(3, hparams.graph_encoder_hidden_dim)

        self.cond_projector = torch.nn.Sequential(
            torch.nn.Linear(6 * hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.graph_encoder_hidden_dim, hparams.code_dim),
        )

    def forward_cond(self, batched_data, cond):
        x, edge_index, edge_attr = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
        )

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layers):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)

        out = self.projector(h)
        out = global_mean_pool(out, batched_data.batch)

        h = torch.cat(
            [
                h,
                self.cond_embedding0(cond[:, 0]),
                self.cond_embedding1(cond[:, 1]),
                self.cond_embedding2(cond[:, 2]),
                self.cond_embedding3(cond[:, 3]),
                self.cond_embedding4(cond[:, 4]),
            ],
            dim=1,
        )

        out0 = self.cond_projector(h)
        out0 = global_mean_pool(out0, batched_data.batch)
        return out, out0

class RelationalGraphDataset(GraphDataset):
    def __init__(self, data_dir, split):
        self.smiles_list = load_smiles_list(data_dir, split)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        pyg_data = smiles2graph(smiles)
        mutate_pyg_data, action_feat = mutate(pyg_data, return_relation=True)

        return pyg_data, mutate_pyg_data, action_feat

    def __len__(self):
        return len(self.smiles_list)

    @staticmethod
    def collate(data_list):
        pyg_data_list, mutate_pyg_data_list, action_feat_list = list(zip(*data_list))
        batched_pyg_data = GraphDataset.collate(pyg_data_list)
        batched_mutate_pyg_data = GraphDataset.collate(mutate_pyg_data_list)
        action_feats = torch.cat(action_feat_list, dim=0)

        return batched_pyg_data, batched_mutate_pyg_data, action_feats


class RelationalAutoEncoder(ContrastiveAutoEncoder):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = RelationalEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)
        self.projector = torch.nn.Linear(hparams.code_dim, hparams.code_dim)

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        batched_input_data0, batched_input_data1, action_feats = batched_input_data
        codes, codes0 = self.encoder.forward_cond(batched_input_data0, action_feats)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics

    def get_input_dataset(self, split):
        return RelationalGraphDataset(self.hparams.data_dir, split)

    def encode(self, batched_input_data):
        return self.encoder(batched_input_data)

    #@staticmethod
    #def collate(data_list):
