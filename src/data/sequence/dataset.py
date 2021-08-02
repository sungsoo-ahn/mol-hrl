import torch
from torch.nn.utils.rnn import pad_sequence

from data.sequence.vocab import PAD_ID
from data.sequence.util import load_vocabulary, load_tokenizer, smiles2sequence
from data.smiles.util import load_smiles_list


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=smiles2sequence):
        super(SequenceDataset, self).__init__()
        self.smiles_list = load_smiles_list(data_dir, split)
        self.tokenizer = load_tokenizer(data_dir)
        self.vocabulary = load_vocabulary(data_dir)
        self.transform = transform

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = self.transform(smiles, self.tokenizer, self.vocabulary)
        return sequence

    def __len__(self):
        return len(self.smiles_list)

    def update(self, smiles_list):
        self.smiles_list.extend(smiles_list)

    @staticmethod
    def collate(data_list):
        lengths = torch.LongTensor([sequence.size(0) for sequence in data_list])
        sequences = pad_sequence(data_list, batch_first=True, padding_value=PAD_ID)
        batched_sequence_data = (sequences, lengths)
        return batched_sequence_data
