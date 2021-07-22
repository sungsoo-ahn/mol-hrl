import torch
from torch_geometric.data import Data
import random

ADD_BOND=1
DELETE_BOND=2
ADD_ATOM=3
DELETE_ATOM=4

MASK_RATE = 0.1

def add_random_edge(edge_index, edge_attr, node0, node1):
    edge_index = torch.cat([edge_index, torch.tensor([[node0, node1], [node1, node0]])], dim=1)
    edge_attr01 = torch.tensor([random.choice(range(6)), random.choice(range(3))]).unsqueeze(0)
    edge_attr = torch.cat([edge_attr, edge_attr01, edge_attr01], dim=0)
    return edge_index, edge_attr, edge_attr01

def mutate(data, return_relation=False):
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
        
    action = random.choice(range(1, 5))
    action_feat = torch.zeros(num_nodes, 5, dtype=torch.long)
    if action == ADD_BOND:
        node0, node1 = random.sample(range(num_nodes), 2)
        edge_index, edge_attr, edge_attr01 = add_random_edge(edge_index, edge_attr, node0, node1)

        action_feat[node0, 0] = ADD_BOND
        action_feat[node1, 0] = ADD_BOND
        action_feat[node0, 3:5] = edge_attr01
        action_feat[node1, 3:5] = edge_attr01

    elif action == DELETE_BOND:
        edge0 = random.choice(range(num_edges))
        node0, node1 = data.edge_index[:, edge0].tolist()
        edge1 = torch.nonzero((data.edge_index[0] == node1) & (data.edge_index[1] == node0)).item()
        
        edge_mask = torch.ones(num_edges, dtype=torch.bool)
        edge_mask[edge0] = False
        edge_mask[edge1] = False

        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask, :]

        action_feat[node0, 0] = DELETE_BOND
        action_feat[node1, 0] = DELETE_BOND
        
    elif action == ADD_ATOM:
        node_feat = torch.tensor([[random.choice(range(120)), random.choice(range(3))]])
        x = torch.cat([x, node_feat], dim=0)
        
        node0 = num_nodes
        node1 = random.choice(range(num_nodes))
        edge_index, edge_attr, edge_attr01 = add_random_edge(edge_index, edge_attr, node0, node1)

        action_feat[node1, 0] = ADD_ATOM
        action_feat[node1, 1:3] = node_feat
        action_feat[node1, 3:5] = edge_attr01

    elif action == DELETE_ATOM:
        node = random.choice(range(num_nodes))
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_mask[node] = False
        x = x[node_mask]

        edge_mask = (data.edge_index[0] != node) & (data.edge_index[1] != node)
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask, :]

        edge_index[edge_index > node] = edge_index[edge_index > node] - 1 

        action_feat[node, 0] = DELETE_ATOM

    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if return_relation:
        return new_data, action_feat
    else:
        return new_data

def mask(data):
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()

    mask_idx = random.sample(range(num_nodes), k=int(MASK_RATE * num_nodes))
    x[mask_idx]= 0

    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_data
    
