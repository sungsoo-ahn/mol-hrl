import torch


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tsrs):
        self.tsrs = tsrs

    def __len__(self):
        return self.tsrs.size(0)

    def __getitem__(self, idx):
        return self.tsrs[idx]

    def collate(self, data_list):
        return torch.stack(data_list, dim=0)


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(self, data_list):
        return [
            dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))
        ]
