from data.smiles.util import load_smiles_list
from data.score.factory import get_scoring_func
import torch
import os


def load_score_list(root_dir, score_func_name, split):
    score_list_path = os.path.join(root_dir, f"{split}_{score_func_name}.pth")

    if not os.path.exists(score_list_path):
        smiles_list = load_smiles_list(root_dir, split)
        _, parallel_score_func, _ = get_scoring_func(score_func_name)
        score_list = parallel_score_func(smiles_list)

        torch.save(score_list, score_list_path)

    with open(score_list_path, "r") as f:
        score_list = torch.load(score_list_path)

    return score_list


def load_scores(root_dir, score_func_name, split):
    score_list = load_score_list(root_dir, score_func_name, split)
    return torch.FloatTensor(score_list).unsqueeze(1)


class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, score_func_names, split):
        super(ScoreDataset, self).__init__()
        # Setup normalization statistics
        train_score_lists = [
            load_score_list(root_dir, score_func_name, "train_labeled")
            for score_func_name in score_func_names
        ]
        train_raw_tsrs = torch.FloatTensor(train_score_lists).T
        self.mean_scores = train_raw_tsrs.mean(dim=0)
        self.std_scores = train_raw_tsrs.std(dim=0)

        # Setup dataset
        score_lists = [
            load_score_list(root_dir, score_func_name, split)
            for score_func_name in score_func_names
        ]
        self.raw_tsrs = torch.FloatTensor(score_lists).T
        self.tsrs = self.normalize(self.raw_tsrs)

    def __len__(self):
        return self.tsrs.size(0)

    def __getitem__(self, idx):
        return self.tsrs[idx]

    @staticmethod
    def collate(data_list):
        return torch.stack(data_list, dim=0)

    def normalize(self, tsrs):
        out = (tsrs - self.mean_scores.unsqueeze(0)) / self.std_scores
        return out
