import torch


class StringDataset(torch.utils.data.Dataset):
    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return self.strings[idx]

    def collate_fn(self, data_list):
        return data_list


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
    def __init__(self, dataset0, dataset1, dataset2=None):
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset0)

    def __getitem__(self, idx):
        if self.dataset2 is None:
            return (self.dataset0[idx], self.dataset1[idx])
        else:
            return (self.dataset0[idx], self.dataset1[idx], self.dataset2[idx])


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
