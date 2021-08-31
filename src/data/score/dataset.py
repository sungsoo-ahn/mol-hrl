from data.util import load_smiles_list, load_score_list
import torch
import os
import numpy as np

def load_statistics(task):
    if task == "plogp":
        mean = 0.024503251342150357
        std = 1.9413246309350511
        success_margin = 0.5
    elif task in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        score_list = load_score_list(task, "train")
        mean = np.mean(score_list)
        std = np.std(score_list)
        print(np.max(score_list), np.min(score_list))
        success_margin = 0.5
    
    return mean, std, success_margin

class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super(ScoreDataset, self).__init__()
        # Setup normalization statistics
        self.scores = torch.FloatTensor(load_score_list(task, split)).T
        self.mean, self.std, _ = load_statistics(task)

    def __len__(self):
        return self.scores.size(0)

    def __getitem__(self, idx):
        return (self.scores[idx] - self.mean) / self.std
    
    def update(self, score_list):
        new_scores = torch.FloatTensor(score_list).view(-1, 1)
        self.raw_tsrs = torch.cat([self.scores, new_scores], dim=0)

    @staticmethod
    def collate(data_list):
        return torch.stack(data_list, dim=0)