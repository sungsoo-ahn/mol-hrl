
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GinConv(MessagePassing):
    def __init__(self, hidden_dim):
        super(GinConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2*hidden_dim), torch.nn.BatchNorm1d(2*hidden_dim), torch.nn.ReLU(), torch.nn.Linear(2*hidden_dim, hidden_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GnnProjector(torch.nn.Module):
    def __init__(self, num_layer, hidden_dim, proj_dim):
        super(GnnProjector, self).__init__()
        self.num_layer = num_layer
        self.atom_encoder = AtomEncoder(hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linear = torch.nn.Linear(hidden_dim, proj_dim)

        for layer in range(num_layer):
            self.convs.append(GinConv(hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        h = self.atom_encoder(x)
        for layer in range(self.num_layer):
            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer - 1:
                h = F.relu(h)

        h = self.linear(h)
        
        return h

        #out = global_add_pool(h, batch)
        #out = self.linear(out)
        #out = F.normalize(out, dim=1, p=2)

