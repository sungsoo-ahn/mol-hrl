import torch
from torch_geometric.data import Data
from ogb.utils.mol import smiles2graph
from rdkit import Chem

def smiles2graph_pyg(smiles):
    graph = smiles2graph(smiles)
    graph_pyg = Data()
    graph_pyg.__num_nodes__ = graph['num_nodes']
    graph_pyg.edge_index = torch.from_numpy(graph['edge_index'])

    del graph['num_nodes']
    del graph['edge_index']

    if graph['edge_feat'] is not None:
        graph_pyg.edge_attr = torch.from_numpy(graph['edge_feat'])
        del graph['edge_feat']

    if graph['node_feat'] is not None:
        graph_pyg.x = torch.from_numpy(graph['node_feat'])
        del graph['node_feat']

    return graph_pyg

def smiles2seq(smiles, tokenizer, vocabulary):
    return torch.tensor(vocabulary.encode(tokenizer.tokenize(smiles)))

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles
