import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from data.graph.dataset import GraphDataset
from data.graph.util import smiles2graph
from data.graph.transform import mask
from data.smiles.util import load_smiles_list
from data.sequence.dataset import SequenceDataset

from module.autoencoder.base import BaseAutoEncoder
from module.autoencoder.contrastive import ContrastiveAutoEncoder
from module.encoder.graph import GraphEncoder 
from module.decoder.sequence import SequenceDecoder

class MaskingEncoder(GraphEncoder):
    def __init__(self, hparams):
        super(MaskingEncoder, self).__init__(hparams)
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

        noderep = self.projector(h)
        graphrep = global_mean_pool(noderep, batched_data.batch)
        return graphrep, noderep

class MaskingAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = MaskingEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)
        self.classifier = torch.nn.Linear(hparams.code_dim, 120)

        self.transform = lambda smiles: mask(smiles2graph(smiles))
        
    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        codes, noderep = self.encoder(batched_input_data)
        
        pred = self.classifier(noderep)
        target = batched_input_data.true_x

        masking_loss = F.cross_entropy(pred, target)
        masking_acc = (torch.argmax(pred, dim=1) == target).float().mean()
        
        loss += masking_loss
        statistics["loss/masking"] = masking_loss
        statistics["acc/masking"] = masking_acc

        return codes, loss, statistics

    def encode(self, batched_input_data):
        return self.encoder(batched_input_data)[0]