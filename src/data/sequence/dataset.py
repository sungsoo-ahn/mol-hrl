from data.sequence.util import Vocabulary
from fsspec.utils import tokenize
import torch
from data.sequence.util import sequence_from_string

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, tokenizer, vocabulary):
        super(SequenceDataset, self).__init__()
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = sequence_from_string(smiles, self.tokenizer, self.vocabulary)
        length = sequence.size(0)

        return sequence, torch.tensor(length)

    def __len__(self):
        return len(self.smiles_list)