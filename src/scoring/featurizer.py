from collections import defaultdict
import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.DataStructs import ConvertToNumpyArray
from guacamol.utils.descriptors import (
    logP,
    qed,
    tpsa,
    bertz,
    mol_weight,
    num_H_donors,
    num_H_acceptors,
    num_rotatable_bonds,
    num_rings,
    num_aromatic_rings,
    num_atoms,
    AtomCounter,
)
from guacamol.utils.fingerprints import get_fingerprint

from joblib import Parallel, delayed


num_F = AtomCounter("F")

get_ecfp4 = lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2)
get_ecfp6 = lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 3)
# get_ap = lambda mol: AllChem.GetAtomPairFingerprint(mol, maxLength=10)

FEATURE_DIM = 5 + 7 + 2 * 2048

FLOAT_FEATURIZERS = [logP, qed, tpsa, bertz, mol_weight]
FLOAT_FEATURE_MEANS = [2.45712358, 0.73184226, 64.8209853, 710.94792876, 332.13967946]
FLOAT_FEATURE_STD = [
    1.43433047e00,
    1.38570162e-01,
    2.29345893e01,
    2.57364488e02,
    6.19439614e01,
]
INT_FEATURIZERS = [
    num_H_donors,
    num_H_acceptors,
    num_rotatable_bonds,
    num_rings,
    num_aromatic_rings,
    num_atoms,
    num_F,
]
INT_FEATURE_MEANS = [
    1.24341768,
    3.96906068,
    4.56015891,
    2.75626964,
    1.84982522,
    43.79362292,
    0.31841287,
]
INT_FEATURE_STDS = [
    0.88174404,
    1.67087038,
    1.55066861,
    1.01224225,
    0.96947708,
    8.49079507,
    0.78181578,
]


def compute_feature(smiles):
    mol = MolFromSmiles(smiles)

    features = defaultdict(list)
    for featurizer, mean, std in zip(FLOAT_FEATURIZERS, FLOAT_FEATURE_MEANS, FLOAT_FEATURE_STD):
        feature = (featurizer(mol) - mean) /std
        features["float"].append(feature)

    features["float"] = np.array(features["float"])

    for featurizer, mean, std in zip(INT_FEATURIZERS, INT_FEATURE_MEANS, INT_FEATURE_STDS):
        feature = (featurizer(mol) - mean) /std
        features["int"].append(feature)

    features["int"] = np.array(features["int"])

    for featurizer in [
        get_ecfp4,
        get_ecfp6,
        # get_ap,
    ]:
        feature = featurizer(mol)
        array = np.zeros((0,), dtype=np.int8)
        ConvertToNumpyArray(feature, array)
        features["fp"].append(array)

    features["fp"] = np.concatenate(features["fp"], axis=0)

    return features