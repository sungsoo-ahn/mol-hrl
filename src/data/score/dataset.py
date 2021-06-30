import torch

class ScoresDataset(torch.utils.data.Dataset):
    def __init__(self, scores_list):
        super(ScoresDataset, self).__init__()
        self.raw_tsrs = torch.FloatTensor(scores_list).T
        self.mean_scores = self.raw_tsrs.mean(dim=0)
        self.std_scores = self.raw_tsrs.std(dim=0)
        self.tsrs = (
            (self.raw_tsrs - self.mean_scores.unsqueeze(0)) / self.std_scores
        )
    
    def __len__(self):
        return self.tsrs.size(0)
    
    def __getitem__(self, idx):
        return self.tsrs[idx]