import torch
from data.graph.dataset import GraphDataset
from data.graph.util import smiles2graph
from data.graph.transform import fragment
from data.sequence.dataset import SequenceDataset, EnumSequenceDataset

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
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]

def load_dataset(dataset_name, task, split):
    if dataset_name == "graph2seq":
        input_dataset = GraphDataset(task, split)
        target_dataset = SequenceDataset(task, split)
        dataset = ZipDataset(input_dataset, target_dataset)
    elif dataset_name == "graph2enumseq":
        input_dataset = GraphDataset(task, split)
        target_dataset = EnumSequenceDataset(task, split)
        dataset = ZipDataset(input_dataset, target_dataset)
    elif dataset_name == "fraggraph2seq":
        input_dataset = GraphDataset(task, split, transform=fragment)
        target_dataset = SequenceDataset(task, split)
        dataset = ZipDataset(input_dataset, target_dataset)

    return dataset

def load_collate(dataset_name):
    if dataset_name in ["graph2seq", "graph2enumseq", "fraggraph2seq"]:
        def collate(data_list):
            input_data_list, target_data_list = zip(*data_list)
            batched_input_data = GraphDataset.collate(input_data_list)
            batched_target_data = SequenceDataset.collate(target_data_list)
            return batched_input_data, batched_target_data
    
    return collate
