from module.encoder.graph import GraphEncoder
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool

from module.autoencoder.base import BaseAutoEncoder
from module.encoder.graph import GraphEncoder
from module.decoder.sequence import SequenceDecoder

class DGIEncoder(GraphEncoder):
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

class DGIAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = DGIEncoder(hparams)
        self.decoder = SequenceDecoder(hparams)
        self.dgi_mat = torch.nn.Parameter(torch.Tensor(hparams.code_dim, hparams.code_dim))
        torch_geometric.nn.inits.uniform(self.dgi_mat.size(0), self.dgi_mat)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dgi_loss_coef", type=float, default=1.0)

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        codes, noderep = self.encoder(batched_input_data)
        graphrep = torch.sigmoid(codes)

        loss, statistics = self.update_dgi_loss(
            noderep, graphrep, batched_input_data.batch, loss, statistics
        )

        return codes, loss, statistics

    def update_dgi_loss(self, noderep, graphrep, batch, loss, statistics):
        positive_expanded_summary_emb = graphrep[batch]

        cycle_index = torch.arange(len(graphrep)) + 1
        cycle_index[-1] = 0
        shifted_summary_emb = graphrep[cycle_index]
        negative_expanded_summary_emb = shifted_summary_emb[batch]

        h = torch.matmul(positive_expanded_summary_emb, self.dgi_mat)
        positive_score = torch.sum(noderep * h, dim=1)

        h = torch.matmul(negative_expanded_summary_emb, self.dgi_mat)
        negative_score = torch.sum(noderep * h, dim=1)

        dgi_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score))
            + F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
        )
        loss += self.hparams.dgi_loss_coef * dgi_loss
        statistics["loss/dgi"] = dgi_loss

        num_corrects = torch.sum(positive_score > 0) + torch.sum(negative_score < 0)
        dgi_acc = num_corrects.float() / float(2 * len(positive_score))
        statistics["acc/dgi"] = dgi_acc

        return loss, statistics

    def encode(self, batched_input_data):
        return self.encoder(batched_input_data)[0]