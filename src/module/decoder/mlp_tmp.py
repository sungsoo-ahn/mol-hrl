import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from data.graph.dataset import GraphDataset
from data.graph.util import smiles2graph

num_atom_type = 1 + 119 + 1  # Frag node (0), atomic (1~119), mask (120)
num_chirality_tag = 3

num_bond_type = 4
num_bond_direction = 3



class MLPDecoder(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim, code_dim):
        super(MLPDecoder, self).__init__()
        self.num_layers = num_layers

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        #self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_dim)
        #self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_dim)

        #torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        #torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.mlps.append(torch.nn.Linear(code_dim, code_dim))
            #self.mlps.append(torch.nn.ReLU(inplace=True))
        self.atom_projector = torch.nn.Linear(code_dim, num_atom_type)
        self.chirality_projector = torch.nn.Linear(code_dim, num_chirality_tag)
        self.edge_projector = torch.nn.Linear(code_dim*2, num_bond_type + 1)


        ###List of batchnorms
        #self.batch_norms = torch.nn.ModuleList()
        #for _ in range(num_layers):
        #    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        #self.projector = torch.nn.Sequential(
        #    torch.nn.Linear(hidden_dim, hidden_dim),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(hidden_dim, code_dim),
        #)

    def forward(self, h, batch):
        #x, edge_index, edge_attr = (
        #    batched_data.x,
        #    batched_data.edge_index,
        #    batched_data.edge_attr,
        #)

        #h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layers):
            h = self.mlps[layer](h)
            #h = self.batch_norms[layer](h)
            #if layer < self.num_layers:
            h = F.relu(h)
        h_atom = self.atom_projector(h)
        #h_chirality = self.chirality_projector(h)
        #counts = torch.bincount(batch)

        #h_adjs = torch.zeros(len(counts), max(counts), max(counts), 2 * h.shape[1]).cuda()
        
        #idx = 0
        
        #for b in range(len(counts)):
        #    h_mol = h[idx : idx + counts[b],:]
        #    
        #    h_mol1 = h_mol.repeat(counts[b],1).reshape(counts[b], counts[b], -1)
        #    h_mol2 = h_mol.repeat_interleave(counts[b], dim=0).reshape(counts[b], counts[b], -1)
        #    h_mol = torch.cat((h_mol1, h_mol2), -1)
        #    h_adjs[b,:h_mol.shape[0], :h_mol.shape[1], :2*h.shape[1]] = h_mol
        #    idx += counts[b]

        #h_edge = self.edge_projector(h_adjs)
        
        #out = self.projector(h)
        #out = global_mean_pool(out, batched_data.batch)

        return h_atom#, h_chirality, h_edge

    def encode_smiles(self, smiles_list, device):
        data_list = [smiles2graph(smiles) for smiles in smiles_list]
        batched_sequence_data = GraphDataset.collate_fn(data_list)
        batched_sequence_data = batched_sequence_data.to(device)
        return self(batched_sequence_data)
