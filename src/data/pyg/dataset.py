import torch
from data.pyg.util import pyg_from_string


class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list):
        super(PyGDataset, self).__init__()
        self.smiles_list = smiles_list

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        pyg_data = pyg_from_string(smiles)

        return pyg_data

    def __len__(self):
        return len(self.smiles_list)
