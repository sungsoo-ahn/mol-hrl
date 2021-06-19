import torch
from rdkit import Chem
from guacamol.utils.chemistry import canonicalize


def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles


def is_valid_smiles(smiles):
    mol = canonicalize(smiles)
    if mol is None or len(mol) == 0:
        return False

    return True
