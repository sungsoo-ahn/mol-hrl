from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from data.graph.dataset import RelationalGraphDataset, ContrastiveGraphDataset

from module.graph.encoder import GraphEncoder
from module.smiles.encoder import SmilesEncoder
from module.selfie.encoder import SelfiesEncoder
from module.smiles.decoder import SmilesDecoder
from module.selfie.decoder import SelfiesDecoder

def get_encoder(hparams):
    encoder_class = {
        "graph": GraphEncoder,
        "smiles": SmilesEncoder,
        "selfie": SelfiesEncoder,
    }[hparams.encoder_type]
    return encoder_class(hparams)

def get_decoder(hparams):
    decoder_class = {
        "smiles": SmilesDecoder,
        "selfie": SelfiesDecoder,
    }[hparams.decoder_type]
    return decoder_class(hparams)


class BaseAutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(BaseAutoEncoder, self).__init__()
        self.hparams = hparams
        self.encoder = get_encoder(hparams)
        self.decoder = get_decoder(hparams)
    
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
        return self.encoder.get_dataset(split)
    
    def get_target_dataset(self, split):
        return self.decoder.get_dataset(split)
    
class ContrastiveAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(ContrastiveAutoEncoder, self).__init__(hparams)
        self.projector = nn.Linear(hparams.code_dim, hparams.code_dim)
    
    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        batched_input_data0, batched_input_data1 = batched_input_data
        codes = codes0 = self.encoder(batched_input_data0)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics
    
    def update_contrastive_loss(self, codes0, codes1, loss, statistics):
        #out0 = F.normalize(self.projector(codes0), p=2, dim=1)
        #out1 = F.normalize(self.projector(codes1), p=2, dim=1)
        #logits = torch.matmul(out0, out1.T)
        logits = -torch.cdist(codes0, codes1, p=2)
        labels = torch.arange(codes0.size(0), device = logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        contrastive_acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        loss += contrastive_loss
        statistics["loss/contrastive"] = contrastive_loss
        statistics["acc/contrastive"] = contrastive_acc
        return loss, statistics

    def get_input_dataset(self, split):
        return ContrastiveGraphDataset(
            self.hparams.data_dir, 
            split,
            smiles_transform_type=self.hparams.input_smiles_transform_type,
            graph_transform_type=self.hparams.input_graph_transform_type
            )

class RelationalAutoEncoder(ContrastiveAutoEncoder):
    def __init__(self, hparams):
        super(RelationalAutoEncoder, self).__init__(hparams)
    
    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        batched_input_data0, batched_input_data1, action_feats = batched_input_data
        codes, codes0 = self.encoder.forward_cond(batched_input_data0, action_feats)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics

    def get_input_dataset(self, split):
        return RelationalGraphDataset(self.hparams.data_dir, split)

class DGIContrastiveAutoEncoder(BaseAutoEncoder):
    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        codes, noderep = self.encoder.forward_reps(batched_input_data)
        logits = -torch.cdist(noderep, codes, p=2)
        targets = batched_input_data.batch
        contrastive_loss = F.cross_entropy(logits, targets)
        contrastive_acc = (torch.argmax(logits, dim=1) == targets).float().mean()
        loss += contrastive_loss
        
        statistics["loss/dgi_contrastive"] = contrastive_loss
        statistics["loss/dgi_acc"] = contrastive_acc

        return codes, loss, statistics

class DGIAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(DGIAutoEncoder, self).__init__(hparams)
        self.dgi_mat = nn.Parameter(torch.Tensor(hparams.code_dim, hparams.code_dim))
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
        positive_score = torch.sum(noderep*h, dim=1) 
        
        h = torch.matmul(negative_expanded_summary_emb, self.dgi_mat)
        negative_score = torch.sum(noderep*h, dim=1)

        dgi_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score)) 
            + F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
        )
        loss += dgi_loss
        statistics["loss/dgi"] = dgi_loss
        
        num_corrects = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0))
        dgi_acc = num_corrects.float() / float(2*len(positive_score))
        statistics["acc/dgi"] = dgi_acc

        return loss, statistics