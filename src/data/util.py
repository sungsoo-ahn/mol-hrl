import torch
import random


def get_pseudorandom_split_idxs(size, split_ratios):
    idxs = list(range(size))
    random.Random(0).shuffle(idxs)

    split_idx = int(size * split_ratios[0])
    train_idxs, vali_idxs = idxs[:split_idx], idxs[split_idx:]

    return train_idxs, vali_idxs


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tsrs):
        self.tsrs = tsrs

    def __len__(self):
        return self.tsrs.size(0)

    def __getitem__(self, idx):
        return self.tsrs[idx]

    def collate_fn(self, data_list):
        return torch.stack(data_list, dim=0)


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset0, dataset1):
        self.dataset0 = dataset0
        self.dataset1 = dataset1

    def __len__(self):
        return len(self.dataset0)

    def __getitem__(self, idx):
        return (self.dataset0[idx], self.dataset1[idx])

    def collate_fn(self, data_list):
        data0_list, data1_list = zip(*data_list)
        batched_data0 = self.dataset0.collate_fn(data0_list)
        batched_data1 = self.dataset1.collate_fn(data1_list)
        return batched_data0, batched_data1


class EnumerateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(idx), self.dataset[idx]

    def collate_fn(self, data_list):
        idxs, data_list = zip(*data_list)
        idxs = torch.stack(idxs, dim=0)
        batched_data = self.dataset.collate_fn(data_list)

        return idxs, batched_data
