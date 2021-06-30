import torch
from data.pyg.util import pyg_from_string
from data.smiles.mutate import mutate


class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, mutate=False):
        super(PyGDataset, self).__init__()
        self.smiles_list = smiles_list
        self.mutate = mutate

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.mutate:
            smiles = mutate(smiles)
            
        pyg_data = pyg_from_string(smiles)

        return pyg_data

    def __len__(self):
        return len(self.smiles_list)
