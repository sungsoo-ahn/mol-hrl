import torch
from torch.nn.utils.rnn import pad_sequence

from data.seq.util import (
    SmilesTokenizer, 
    load_vocabulary, 
    load_tokenizer, 
    sequence_from_string, 
    PAD_ID,
    MASK_ID,
)
from data.smiles.util import randomize_smiles, load_smiles_list
from data.selfies.mutate import mutate

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, use_random_smiles=False, mask_rate=0.0, mutate=False):
        super(SequenceDataset, self).__init__()
        self.smiles_list = load_smiles_list(data_dir, split)
        self.tokenizer = load_tokenizer(data_dir)
        self.vocabulary = load_vocabulary(data_dir)
        self.use_random_smiles = use_random_smiles
        self.mask_rate = mask_rate
        self.mutate = mutate

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.mutate:
            smiles = mutate(smiles)

        if self.use_random_smiles:
            smiles = randomize_smiles(smiles)

        sequence = sequence_from_string(smiles, self.tokenizer, self.vocabulary)
        if self.mask_rate > 0.0:
            sequence = mask_sequence(sequence, self.mask_rate)

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