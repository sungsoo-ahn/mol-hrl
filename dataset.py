import pandas as pd
import shutil, os
import numpy as np
import os.path as osp
import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch

from util.mol import smiles2seq, smiles2graph_pyg
from vocabulary import PAD_TOKEN

PADDING_VALUE=0

def pyg_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return Batch.from_data_list(batch, [], [])
    elif isinstance(elem, torch.Tensor):
        return default_collate(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: pyg_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return type(elem)(*(pyg_collate(s) for s in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        return [pyg_collate(s) for s in zip(*batch)]

    raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, tokenizer, vocabulary, return_seq, return_graph, transform, aug_transform):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.return_seq = return_seq
        self.return_graph = return_graph
        self.transform = transform
        self.aug_transform = aug_transform

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.transform is not None:
            smiles = self.transform(smiles)

        if self.aug_transform is not None:
            aug_smiles = self.aug_transform(smiles)

        out = []
        if self.return_seq:
            seq = smiles2seq(smiles, self.tokenizer, self.vocabulary)
            aug_seq = smiles2seq(aug_smiles, self.tokenizer, self.vocabulary)
            out.append(seq)
            out.append(aug_seq)
        
        if self.return_graph:
            graph = smiles2graph_pyg(smiles)
            aug_graph = smiles2graph_pyg(aug_smiles)
            out.append(graph)
            out.append(aug_graph)

        return out

    def __len__(self):
        return len(self.smiles_list)

    def collate_fn(self, elems):
        elems = list(zip(*elems))
        out = []
        if self.return_seq:
            seqs = elems.pop(0)
            aug_seqs = elems.pop(0)
            
            lengths = torch.tensor([seq.size(0) for seq in seqs])
            aug_lengths = torch.tensor([seq.size(0) for seq in aug_seqs])
            
            seqs = pad_sequence(seqs, batch_first=True, padding_value=PADDING_VALUE)
            aug_seqs = pad_sequence(aug_seqs, batch_first=True, padding_value=PADDING_VALUE)
            
            out.append(seqs)
            out.append(aug_seqs)
            out.append(lengths)
            out.append(aug_lengths)
        
        if self.return_graph:
            graphs = elems.pop(0)
            aug_graphs = elems.pop(0)
            graphs = pyg_collate(graphs)
            aug_graphs = pyg_collate(aug_graphs)
            out.append(graphs)
            out.append(aug_graphs)

        return out