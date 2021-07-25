import torch
import torch.nn.functional as F
import torch_geometric

from module.autoencoder.base import BaseAutoEncoder


class MaskingAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(MaskingAutoEncoder, self).__init__(hparams)
        self.dgi_mat = torch.nn.Parameter(torch.Tensor(hparams.code_dim, hparams.code_dim))
        torch_geometric.nn.inits.uniform(self.dgi_mat.size(0), self.dgi_mat)

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        codes, noderep = self.encoder.forward_reps(batched_input_data)
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
        loss += dgi_loss
        statistics["loss/dgi"] = dgi_loss

        num_corrects = torch.sum(positive_score > 0) + torch.sum(negative_score < 0)
        dgi_acc = num_corrects.float() / float(2 * len(positive_score))
        statistics["acc/dgi"] = dgi_acc

        return loss, statistics
