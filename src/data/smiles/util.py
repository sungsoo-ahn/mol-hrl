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


def load_smiles_list(dir):
    smiles_list_path = os.path.join(dir, "smiles_list.txt")
    with open(smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    return smiles_list


def load_split_smiles_list(dir, score_func_name=None):
    smiles_list = load_smiles_list(dir)

    train_idxs_path = os.path.join(dir, "train_idxs.pth")
    vali_idxs_path = os.path.join(dir, "vali_idxs.pth")

    if not os.path.exists(train_idxs_path) or not os.path.exists(vali_idxs_path):
        train_idxs, vali_idxs = get_pseudorandom_split_idxs(len(smiles_list), [0.95, 0.05])
        torch.save(train_idxs, train_idxs_path)
        torch.save(vali_idxs, vali_idxs_path)

    with open(train_idxs_path, "r") as f:
        train_idxs = torch.load(train_idxs_path)

    with open(vali_idxs_path, "r") as f:
        vali_idxs = torch.load(vali_idxs_path)

    train_smiles_list = [smiles_list[idx] for idx in train_idxs]
    vali_smiles_list = [smiles_list[idx] for idx in vali_idxs]

    if score_func_name is None:
        return train_smiles_list, vali_smiles_list

    train_score_list_path = os.path.join(dir, "train_{score_func_name}.pth")
    vali_score_list_path = os.path.join(dir, "vali_{score_func_name}.pth")

    if not os.path.exists(train_score_list_path) or not os.path.exists(vali_score_list_path):
        elem_score_func, parallel_score_func = get_scoring_func(score_func_name)
        train_score_list = parallel_score_func(train_smiles_list)
        vali_score_list = parallel_score_func(vali_smiles_list)

        torch.save(train_score_list, train_score_list_path)
        torch.save(vali_score_list, vali_score_list_path)

    with open(train_score_list_path, "r") as f:
        train_score_list = torch.load(train_score_list_path)

    with open(vali_score_list_path, "r") as f:
        vali_score_list = torch.load(vali_score_list_path)

    return (train_smiles_list, vali_smiles_list), (train_score_list, vali_score_list)


def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles


def is_valid_smiles(smiles):
    mol = canonicalize(smiles)
    if mol is None or len(mol) == 0:
        return False

    return True
