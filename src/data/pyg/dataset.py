import torch

class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, pyg_handler):
        super(PyGDataset, self).__init__()
        self.smiles_list = smiles_list
        self.handler = pyg_handler

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("data")
        group.add_argument("--data_root", type=str, default="../resource/data/zinc")
        return parser

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        pyg_data = self.handler.pyg_from_string(smiles)

        return pyg_data

    def __len__(self):
        return len(self.smiles_list)