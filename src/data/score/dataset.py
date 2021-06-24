import torch
from data.smiles.util import load_split_smiles_list
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super(SequenceDataset, self).__init__()
        train_smiles_list, vali_smiles_list = load_split_smiles_list(dir)
        


