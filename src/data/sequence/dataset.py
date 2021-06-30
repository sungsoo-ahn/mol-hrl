from data.sequence.util import SmilesTokenizer, create_vocabulary
import torch
from data.sequence.util import sequence_from_string
from data.smiles.util import randomize_smiles

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, randomize_smiles=False):
        super(SequenceDataset, self).__init__()
        self.smiles_list = smiles_list
        self.tokenizer = SmilesTokenizer()
        self.vocabulary = create_vocabulary(smiles_list, self.tokenizer)
        self.randomize_smiles = randomize_smiles

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.randomize_smiles:
            smiles = randomize_smiles(smiles)

        sequence = sequence_from_string(smiles, self.tokenizer, self.vocabulary)
        length = sequence.size(0)

        return sequence, torch.tensor(length)

    def __len__(self):
        return len(self.smiles_list)
