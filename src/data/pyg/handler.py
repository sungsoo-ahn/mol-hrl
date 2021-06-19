import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

# allowable node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def get_atom_feature(atom_type):
    atom = Chem.Atom(atom_type)
    atom_feature = [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())] + [
        allowable_features["possible_chirality_list"].index(atom.GetChiralTag())
    ]
    atom_feature = torch.tensor(np.array([atom_feature]), dtype=torch.long)
    return atom_feature


def get_bond_feature(bond_type):
    edge_feature = [bond_type - 1] + [0]
    edge_feature = torch.tensor(np.array([edge_feature]), dtype=torch.long)
    return edge_feature


class PyGHandler:
    def pyg_from_string(self, string):
        mol = Chem.MolFromSmiles(string)

        # atoms
        num_atom_features = 2  # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())
            ] + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [allowable_features["possible_bonds"].index(bond.GetBondType())] + [
                    allowable_features["possible_bond_dirs"].index(bond.GetBondDir())
                ]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        else:  # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        pyg = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return pyg

    def string_from_pyg(self, pyg):
        data_x, data_edge_index, data_edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr

        mol = Chem.RWMol()

        # atoms
        atom_features = data_x.cpu().numpy()
        num_atoms = atom_features.shape[0]
        for i in range(num_atoms):
            atomic_num_idx, chirality_tag_idx = atom_features[i]
            atomic_num = allowable_features["possible_atomic_num_list"][atomic_num_idx]
            chirality_tag = allowable_features["possible_chirality_list"][chirality_tag_idx]
            atom = Chem.Atom(atomic_num)
            atom.SetChiralTag(chirality_tag)
            mol.AddAtom(atom)

        # bonds
        edge_index = data_edge_index.cpu().numpy()
        edge_attr = data_edge_attr.cpu().numpy()
        num_bonds = edge_index.shape[1]
        for j in range(0, num_bonds, 2):
            begin_idx = int(edge_index[0, j])
            end_idx = int(edge_index[1, j])
            bond_type_idx, bond_dir_idx = edge_attr[j]
            bond_type = allowable_features["possible_bonds"][bond_type_idx]
            bond_dir = allowable_features["possible_bond_dirs"][bond_dir_idx]
            mol.AddBond(begin_idx, end_idx, bond_type)
            # set bond direction
            new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
            new_bond.SetBondDir(bond_dir)

        return mol