import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from data.graph.dataset import GraphDataset
from data.graph.util import pyg_from_string

num_atom_type = 120
num_chirality_tag = 3

num_bond_type = 6
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = "add"

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.to(torch.long)

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super(GraphEncoder, self).__init__()
        self.num_layers = hparams.graph_encoder_num_layers

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hparams.graph_encoder_hidden_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hparams.graph_encoder_hidden_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(hparams.graph_encoder_num_layers):
            self.gnns.append(GINConv(hparams.graph_encoder_hidden_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(hparams.graph_encoder_num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(hparams.graph_encoder_hidden_dim))

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hparams.graph_encoder_hidden_dim, hparams.graph_encoder_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.graph_encoder_hidden_dim, hparams.code_dim),
        )

        self.cond_embedding = torch.nn.Embedding(5, hparams.code_dim)
        

    def forward(self, batched_data):
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
        return out
    
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
    
        out0 = self.projector(h + self.cond_embedding(cond))
        out0 = global_mean_pool(out0, batched_data.batch)
        
        return out, out0
    
    def encode_smiles(self, smiles_list, device):
        data_list = [pyg_from_string(smiles) for smiles in smiles_list]
        batched_sequence_data = GraphDataset.collate_fn(data_list)
        batched_sequence_data = batched_sequence_data.to(device)
        return self(batched_sequence_data)
