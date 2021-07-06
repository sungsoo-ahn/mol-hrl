import os
import random
import torch

from data.score.factory import get_scoring_func


def load_raw_data(root_dir, score_func_names, train_ratio, label_ratio):
    smiles_list_path = os.path.join(root_dir, "smiles_list.txt")
    with open(smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    scores_list_path = os.path.join(root_dir, "scores_list.pth")
    if os.path.exists(scores_list_path):
        with open(scores_list_path, "rb") as f:
            scores_list = torch.load(f)
    else:
        scores_list = []
        for name in score_func_names:
            _, parallel_score_func, corrupt_score = get_scoring_func(
                name, num_workers=32
            )
            scores = parallel_score_func(smiles_list)
            if corrupt_score in scores:
                assert False

            scores_list.append(scores)

        torch.save(scores_list, scores_list_path)

    num_samples = len(smiles_list)
    idxs = list(range(num_samples))
    random.Random(0).shuffle(idxs)

    idx0 = int(label_ratio * num_samples)
    idx1 = int(train_ratio * num_samples)
    split_idxs = {
        "train": idxs[:idx1],
        "val": idxs[idx1:],
        "train_labeled": idxs[:idx0],
    }

    return smiles_list, scores_list, split_idxs


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate_fn(self, data_list):
        return [
            dataset.collate_fn(data_list)
            for dataset, data_list in zip(self.datasets, zip(*data_list))
        ]
