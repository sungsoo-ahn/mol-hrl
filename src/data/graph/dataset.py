import torch
import torch_geometric
from data.graph.util import smiles2graph
from data.util import load_smiles_list


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, task, split, transform=smiles2graph):
        super(GraphDataset, self).__init__()
        self.smiles_list = load_smiles_list(task, split)
        self.transform = transform

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        graph = self.transform(smiles)
        return graph

    def __len__(self):
        return len(self.smiles_list)

    @staticmethod
    def collate(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = torch_geometric.data.Data()
        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_size = len(data_list)

        cumsum_node = 0
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node

                batch[key].append(item)

            cumsum_node += num_nodes

        batch.batch = torch.cat(batch.batch, dim=-1)

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch = batch.contiguous()

        return batch

    def update(self, smiles_list):
        self.smiles_list.extend(smiles_list)
