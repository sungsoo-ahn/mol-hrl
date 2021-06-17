from collections import defaultdict
import numpy as np
from scoring.featurizer import compute_feature
from joblib import Parallel, delayed

def compute_dataset_features(dir, tag):
    with open(f"{dir}/{tag}.txt", "r") as f:
        smiles_list = f.read().splitlines()

    features = Parallel(n_jobs=8)(
        delayed(compute_feature)(smiles) for smiles in smiles_list
    )

    return features


if __name__ == "__main__":
    from tqdm import tqdm

    features = compute_dataset_features("../resource/data", "zinc")
    
    int_mean = np.mean(np.stack([feature["int"] for feature in features], axis=0), axis=0)
    int_std = np.std(np.stack([feature["int"] for feature in features], axis=0), axis=0)
    print(int_mean)
    print(int_std)
    
    float_mean = np.mean(np.stack([feature["float"] for feature in features], axis=0), axis=0)
    float_std = np.std(np.stack([feature["float"] for feature in features], axis=0), axis=0)
    print(float_mean)
    print(float_std)