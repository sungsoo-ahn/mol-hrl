import torch
import torch_geometric
from data.graph.util import pyg_from_string
from data.selfies.mutate import mutate
from data.smiles.util import load_smiles_list, randomize_smiles
from data.graph.transform import pyg_mutate, pyg_mask


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, smiles_transform_type, graph_transform_type):
        super(GraphDataset, self).__init__()
        self.smiles_list = load_smiles_list(data_dir, split)
        
        if smiles_transform_type == "none":
            self.smiles_transform = lambda smiles: smiles
        elif smiles_transform_type == "randomize_order":
            self.smiles_transform = randomize_smiles
            print("warning smiles randomization is meaningless")
        
        if graph_transform_type == "none":
            self.graph_transform = lambda graph: graph
        elif graph_transform_type == "mask":
            self.graph_transform = pyg_mask
        elif graph_transform_type == "mutate":
            self.graph_transform = lambda graph: pyg_mutate(graph)[0]

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        smiles = self.smiles_transform(smiles)
        graph = pyg_from_string(smiles)
        graph = self.graph_transform(graph)
        return graph 

    def __len__(self):
        return len(self.smiles_list)

    @staticmethod
    def collate_fn(data_list):
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

class ContrastiveGraphDataset(GraphDataset):
    def __getitem__(self, idx):
        smiles0 = self.smiles_list[idx]
        smiles1 = self.smiles_transform(smiles0)
        
        graph0 = pyg_from_string(smiles0)
        graph1 = pyg_from_string(smiles1)

        graph1 = self.graph_transform(graph1)
        return graph0, graph1
    
    @staticmethod
    def collate_fn(data_list):
        pyg_data_list0, pyg_data_list1 = list(zip(*data_list))
        batched_pyg_data0 = (
            super(ContrastiveGraphDataset, ContrastiveGraphDataset).collate_fn(pyg_data_list0)
        )
        batched_pyg_data1 = (
            super(ContrastiveGraphDataset, ContrastiveGraphDataset).collate_fn(pyg_data_list1)
        )
        
        return batched_pyg_data0, batched_pyg_data1
        
class RelationalGraphDataset(GraphDataset):
    def __init__(self, data_dir, split):
        self.smiles_list = load_smiles_list(data_dir, split)
        
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        pyg_data = pyg_from_string(smiles)
        mutate_pyg_data, action_feat = pyg_mutate(pyg_data)

        return pyg_data, mutate_pyg_data, action_feat

    def __len__(self):
        return len(self.smiles_list)

    @staticmethod
    def collate_fn(data_list):
        pyg_data_list, mutate_pyg_data_list, action_feat_list = list(zip(*data_list))
        batched_pyg_data = super(RelationalGraphDataset, RelationalGraphDataset).collate_fn(pyg_data_list)
        batched_mutate_pyg_data = super(RelationalGraphDataset, RelationalGraphDataset).collate_fn(mutate_pyg_data_list)
        action_feats = torch.cat(action_feat_list, dim=0)
        
        return batched_pyg_data, batched_mutate_pyg_data, action_feats
        