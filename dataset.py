import torch
from torch.nn.utils.rnn import pad_sequence

from vocabulary import PAD_TOKEN, smiles2seq

PADDING_VALUE=0

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, tokenizer, vocabulary, transform):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.transform is not None:
            smiles = self.transform(smiles)

        seq = smiles2seq(smiles, self.tokenizer, self.vocabulary)

        return seq

    def __len__(self):
        return len(self.smiles_list)

    def collate_fn(self, seqs):
        lengths = torch.tensor([seq.size(0) for seq in seqs])
        seqs = pad_sequence(seqs, batch_first=True, padding_value=PADDING_VALUE)
        
        return seqs, lengths

class PairedSmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, tokenizer, vocabulary, transform):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, idx):
        smiles0 = smiles1 = self.smiles_list[idx]
        if self.transform is not None:
            smiles0 = self.transform(smiles0)
            smiles1 = self.transform(smiles1)

        seq0 = smiles2seq(smiles0, self.tokenizer, self.vocabulary)
        seq1 = smiles2seq(smiles1, self.tokenizer, self.vocabulary)

        return seq0, seq1

    def __len__(self):
        return len(self.smiles_list)

    def collate_fn(self, zipped_seqs):
        seqs0, seqs1 = zip(*zipped_seqs)
        lengths0 = torch.tensor([seq.size(0) for seq in seqs0])
        lengths1 = torch.tensor([seq.size(0) for seq in seqs1])
        seqs0 = pad_sequence(seqs0, batch_first=True, padding_value=PADDING_VALUE)
        seqs1 = pad_sequence(seqs1, batch_first=True, padding_value=PADDING_VALUE)
        
        return seqs0, seqs1, lengths0, lengths1
