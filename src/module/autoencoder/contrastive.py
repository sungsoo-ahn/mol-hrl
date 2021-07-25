from data.sequence.dataset import SequenceDataset
import torch
import torch.nn.functional as F
from module.autoencoder.base import BaseAutoEncoder
from data.graph.dataset import GraphDataset
from data.graph.util import smiles2graph
from data.graph.transform import mask, fragment_contract, subgraph

class ContrastiveAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(ContrastiveAutoEncoder, self).__init__(hparams)
        self.projector = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        
        if hparams.input_graph_fragment_contract:
            self.transform = fragment_contract
        elif hparams.input_graph_mask:
            self.transform = lambda smiles: mask(smiles2graph(smiles))
        elif hparams.input_graph_subgraph:
            self.transform = subgraph
        

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        batched_input_data0, batched_input_data1 = batched_input_data
        codes = codes0 = self.encoder(batched_input_data0)
        codes1 = self.encoder(batched_input_data1)
        loss, statistics = self.update_contrastive_loss(codes0, codes1, loss, statistics)
        return codes, loss, statistics

    def update_contrastive_loss(self, codes0, codes1, loss, statistics):
        out0 = F.normalize(self.projector(codes0), p=2, dim=1)
        out1 = F.normalize(self.projector(codes1), p=2, dim=1)
        logits = torch.matmul(out0, out1.T)
        labels = torch.arange(codes0.size(0), device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        contrastive_acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        loss += contrastive_loss
        statistics["loss/contrastive"] = contrastive_loss
        statistics["acc/contrastive"] = contrastive_acc
        return loss, statistics

    def get_input_dataset(self, split):
        def transform(data):
            data0, data1 = smiles2graph(data), self.transform(data)
            return data0, data1

        return GraphDataset(self.hparams.data_dir, split, transform=transform)

    @staticmethod
    def collate(data_list):
        input_data_list, target_data_list = zip(*data_list)
        input_data_list0, input_data_list1 = zip(*input_data_list)
        batched_input_data0 = GraphDataset.collate(input_data_list0)
        batched_input_data1 = GraphDataset.collate(input_data_list1)
        batched_target_data = SequenceDataset.collate(target_data_list)
        return (batched_input_data0, batched_input_data1), batched_target_data