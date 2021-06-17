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


def compute_feature(smiles):
    mol = MolFromSmiles(smiles)

    features = defaultdict(list)
    for featurizer in [
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
        num_F,
    ]:
        feature = featurizer(mol)
        features["float"].append(feature)

    features["float"] = np.array(features["float"])

    for featurizer in [
        num_H_donors,
        num_H_acceptors,
        num_rotatable_bonds,
        num_rings,
        num_aromatic_rings,
        num_atoms,
        num_F,
    ]:
        feature = featurizer(mol)
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


def create_scored_dataset(dir, tag):
    with open(f"{dir}/{tag}.txt", "r") as f:
        smiles_list = f.read().splitlines()

    features = Parallel(n_jobs=8)(
        delayed(compute_feature)(smiles) for smiles in smiles_list
    )

    return features


if __name__ == "__main__":
    from tqdm import tqdm

    for _ in tqdm(range(10000)):
        features = compute_feature("CCCCC")

    print(features)
