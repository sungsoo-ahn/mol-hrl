from rdkit import Chem

import torch
from torch.nn.utils.rnn import pad_sequence

from data.sequence.vocab import PAD_ID
from data.sequence.util import sequence2smiles, smiles2sequence
from data.util import load_smiles_list, load_vocabulary, load_tokenizer


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, task, split, transform=smiles2sequence):
        super(SequenceDataset, self).__init__()
        self.smiles_list = load_smiles_list(task, split)
        self.tokenizer = load_tokenizer()
        self.vocabulary = load_vocabulary()
        self.transform = transform

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = torch.LongTensor(smiles2sequence(smiles, self.tokenizer, self.vocabulary))
        return sequence

    def __len__(self):
        return len(self.smiles_list)

    def update(self, smiles_list):
        self.smiles_list.extend(smiles_list)

    @staticmethod
    def collate(data_list):
        sequences = pad_sequence(data_list, batch_first=True, padding_value=PAD_ID)
        return sequences


def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles

class EnumSequenceDataset(SequenceDataset):
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        smiles = randomize_smiles(smiles)
        sequence = torch.LongTensor(self.transform(smiles, self.tokenizer, self.vocabulary))
        return sequence