from data.util import load_smiles_list, load_score_list
import torch
import os
import numpy as np

PLOGP_MEAN = -0.028084750892405044
PLOGP_STD = 2.0570724640259397
PLOGP_SUCCESS_MARGIN = 0.5

class PLogPDataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super(PLogPDataset, self).__init__()
        # Setup normalization statistics
        self.scores = torch.FloatTensor(load_score_list("plogp", None)).T
        
    def __len__(self):
        return self.scores.size(0)

    def __getitem__(self, idx):
        #noise = np.random.uniform(-PLOGP_SUCCESS_MARGIN, PLOGP_SUCCESS_MARGIN)
        return (self.scores[idx] - PLOGP_MEAN) / PLOGP_STD
    
    def update(self, score_list):
        new_scores = torch.FloatTensor(score_list).view(-1, 1)
        self.raw_tsrs = torch.cat([self.scores, new_scores], dim=0)

    @staticmethod
    def collate(data_list):
        return torch.stack(data_list, dim=0)