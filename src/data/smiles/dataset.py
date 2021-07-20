import torch
from torch.nn.utils.rnn import pad_sequence

from data.smiles.vocab import (
    load_vocabulary, 
    load_tokenizer, 
    sequence_from_string, 
    PAD_ID,
    MASK_ID,
)
from data.smiles.util import load_smiles_list
from data.smiles.transform import randomize

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, smiles_transform_type="none", seq_transform_type="none"):
        super(SmilesDataset, self).__init__()
        self.smiles_list = load_smiles_list(data_dir, split)
        self.tokenizer = load_tokenizer(data_dir)
        self.vocabulary = load_vocabulary(data_dir)

        if smiles_transform_type == "none":
            self.smiles_transform = lambda smiles: smiles
        elif smiles_transform_type == "randomize_order":
            self.smiles_transform = randomize
            
        if seq_transform_type == "none":
            self.seq_transform = lambda seq: seq
        elif seq_transform_type == "mask":
            self.seq_transform = lambda seq: mask_sequence(seq, 0.1)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        smiles = self.smiles_transform(smiles)
        
        sequence = sequence_from_string(smiles, self.tokenizer, self.vocabulary)
        sequence = self.seq_transform(sequence)
        length = sequence.size(0)

        return sequence, torch.tensor(length)

    def __len__(self):
        return len(self.smiles_list)

    @staticmethod
    def collate_fn(data_list):
        sequences, lengths = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
        lengths = torch.stack(lengths)
        batched_sequence_data = (sequences, lengths)
        return batched_sequence_data

def mask_sequence(sequence, mask_rate):
    mask = torch.bernoulli(torch.full((sequence.size(0),), mask_rate)).bool()
    mask[0] = False
    mask[-1] = False
    sequence[mask] = MASK_ID
    return sequence