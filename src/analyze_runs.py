import pandas as pd
import os, sys
from collections import defaultdict
import neptune.new as neptune

import rdkit.Chem as Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from data.score.factory import get_scoring_func

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

RUN_NAMES = [
    "MOLREP-223", 
    ]
SCORING_FUNC_NAME = "logp"
MARKERS = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']

def is_pareto_efficient(scores):
    costs = -scores
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    
    is_efficient_mask = np.zeros(n_points, dtype = bool)
    is_efficient_mask[is_efficient] = True
    return scores[is_efficient_mask, :]
    
if __name__ == "__main__":
    run_results = defaultdict(dict) 
    for run_name in RUN_NAMES:
        run = neptune.init(
            project="sungsahn0215/molrep",
            run=run_name
            )
        run["gradopt_linear_penalized_logp_smiles_traj"].download("tmp.csv")
        run_results[run_name]["smiles_raw"] = pd.read_csv("tmp.csv")
        

    # Compute actual scores
    _, scoring_func, corrupt_score = get_scoring_func(SCORING_FUNC_NAME)
    for run_name in RUN_NAMES:
        seen_smiles = []
        for idx in range(1024):
            seen_smiles.extend(run_results[run_name]["smiles_raw"][f"{idx}"])
        
        seen_smiles = list(set(seen_smiles))
        scores = scoring_func(seen_smiles)
        
        smiles_list = [
            smiles for score, smiles in zip(scores, seen_smiles) if score > corrupt_score + 1e-3
            ]
        scores_list = [score for score in scores if score > corrupt_score + 1e-3]

        run_results[run_name]["smiles"] = smiles_list
        run_results[run_name][SCORING_FUNC_NAME] = scores_list

    # Compute sascore
    sa_scoring_func = lambda smiles: sascorer.calculateScore(Chem.MolFromSmiles(smiles))
    qed_scoring_func = get_scoring_func("qed")[0]
    for run_name in RUN_NAMES:
        smiles_list = run_results[run_name]["smiles"]
        run_results[run_name]["sa"] = [sa_scoring_func(smiles) for smiles in smiles_list]
        run_results[run_name]["qed"] = [qed_scoring_func(smiles) for smiles in smiles_list]
        
    for run_name, marker in zip(RUN_NAMES, MARKERS):
        smiles_list = run_results[run_name]["smiles"]
        score_list = run_results[run_name][SCORING_FUNC_NAME]
        sa_list = run_results[run_name]["sa"]
        qed_list = run_results[run_name]["qed"]

        plt.plot(np.array(score_list), np.array(sa_list), MARKERS[-1], label=run_name)

        #pareto_pts = is_pareto_efficient(np.array([score_list, qed_list]).T)
        #plt.plot(pareto_pts[:, 0], pareto_pts[:, 1], marker, label=run_name)
    
    plt.legend(numpoints=1)
    plt.savefig("./tmp.png")