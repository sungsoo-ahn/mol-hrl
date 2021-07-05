from data.score.factory import get_scoring_func
import os
import random
import torch
from rdkit import Chem
from guacamol.utils.chemistry import canonicalize


def get_pseudorandom_split_idxs(size, split_ratios):
    idxs = list(range(size))
    random.Random(0).shuffle(idxs)

    split_idx = int(size * split_ratios[0])
    train_idxs, vali_idxs = idxs[:split_idx], idxs[split_idx:]

    return train_idxs, vali_idxs

def load_smiles_list(root_dir, split):
    smiles_list_path = os.path.join(root_dir, "smiles_list.txt")
    with open(smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()
    
    train_idxs_path = os.path.join(root_dir, "train_idxs.pth")
    vali_idxs_path = os.path.join(root_dir, "vali_idxs.pth")

    if not os.path.exists(train_idxs_path) or not os.path.exists(vali_idxs_path):
        train_idxs, vali_idxs = get_pseudorandom_split_idxs(len(smiles_list), [0.9, 0.1])
        torch.save(train_idxs, train_idxs_path)
        torch.save(vali_idxs, vali_idxs_path)

    with open(train_idxs_path, "r") as f:
        train_idxs = torch.load(train_idxs_path)

    with open(vali_idxs_path, "r") as f:
        vali_idxs = torch.load(vali_idxs_path)

    if split == "full":
        return smiles_list
    elif split == "train":
        return [smiles_list[idx] for idx in train_idxs]
    elif split == "val":
        return [smiles_list[idx] for idx in vali_idxs]
    elif split == "train_labeled":
        return [smiles_list[idx] for idx in train_idxs[:int(len(smiles_list) * 0.1)]]


def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles


def is_valid_smiles(smiles):
    mol = canonicalize(smiles)
    if mol is None or len(mol) == 0:
        return False

    return True
