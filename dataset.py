import torch
from torch.nn.utils.rnn import pad_sequence

from vocabulary import PAD_TOKEN

class Smiles2SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, tokenizer, vocabulary):
        self.smiles_list = smiles_list
        self.tsrs = [
            torch.tensor(vocabulary.encode(tokenizer.tokenize(smiles)))
            for smiles in smiles_list
            ]
        self.padding_value = 0

    def __getitem__(self, idx):
        return self.tsrs[idx]

    def __len__(self):
        return len(self.tsrs)

    @staticmethod
    def collate_fn(tsrs):
        lengths = torch.tensor([tsr.size(0) for tsr in tsrs])
        padded_tsrs = pad_sequence(tsrs, batch_first=True, padding_value=0)
        return padded_tsrs, lengths