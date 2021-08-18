import random

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import BRICS

from data.graph.util import smiles2graph, mol2graph

def mask(smiles, mask_rate=0.3):
    data = smiles2graph(smiles)
    num_nodes = data.x.size(0)
    k = random.choice(range(int(mask_rate * num_nodes)))
    if k == 0:
        return data
    
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    x[random.sample(range(num_nodes), k=k), 0] = 0
    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_data

def fragment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    k = random.randint(0, len(brics_bonds))
    if k == 0:
        return mol2graph(mol)
    
    brics_bonds = random.sample(brics_bonds, k=k)
    fragged_mol = BRICS.BreakBRICSBonds(mol, bonds=brics_bonds)
    return mol2graph(fragged_mol)