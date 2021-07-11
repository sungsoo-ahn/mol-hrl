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

RUN_NAMES = ["MOLREP-63"]
SCORING_FUNC_NAME = "penalized_logp"
MARKERS = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

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
        smiles_log = run["smiles"].fetch_values()["value"].tolist()
        smiles_log = [text.split(",") for text in smiles_log]
        run_results[run_name]["smiles_log"] = smiles_log

    # Compute actual scores
    _, scoring_func, corrupt_score = get_scoring_func(SCORING_FUNC_NAME)
    for run_name in RUN_NAMES:
        scores_log = []
        for smiles_list in run_results[run_name]["smiles_log"]:
            score_list = scoring_func(smiles_list)
            for idx, score in enumerate(score_list):
                if score < corrupt_score + 1e-3:
                    score_list[idx] = None
            
            scores_log.append(score_list)

        run_results[run_name][SCORING_FUNC_NAME] = scores_log

    # Compute sascore
    sa_scoring_func = lambda smiles: sascorer.calculateScore(Chem.MolFromSmiles(smiles))
    qed_scoring_func = get_scoring_func("qed")[0]
    for run_name in RUN_NAMES:
        smiles_log = run_results[run_name]["smiles_log"]
        score_log = run_results[run_name][SCORING_FUNC_NAME]
        sa_log, qed_log = [], []
        for smiles_list, score_list in zip(smiles_log, score_log):
            sa_list, qed_list = [], []
            for smiles, score in zip(smiles_list, score_list):
                if score is None:
                    sa_list.append(None)
                    qed_list.append(None)
                else:
                    sa_list.append(sa_scoring_func(smiles))
                    qed_list.append(qed_scoring_func(smiles))

            sa_log.append(sa_list)
            qed_log.append(qed_list)
        
        run_results[run_name]["sa"] = sa_log
        run_results[run_name]["qed"] = qed_log

    def unravel_list(target_log):
        target_log = [x for xx in target_log for x in xx]
        return target_log

    for run_name, marker in zip(RUN_NAMES, MARKERS):
        smiles_log = unravel_list(run_results[run_name]["smiles_log"]) 
        score_log = unravel_list(run_results[run_name][SCORING_FUNC_NAME])
        sa_log = unravel_list(run_results[run_name]["sa"])
        qed_log = unravel_list(run_results[run_name]["qed"])

        score_list = []
        sa_list = []
        qed_list = []
        for smiles, score, sa, qed in zip(smiles_log, score_log, sa_log, qed_log):
            if score is None:
                continue

            score_list.append(score)
            sa_list.append(sa)
            qed_list.append(qed)

        plt.plot(np.array(score_list), np.array(sa_list), MARKERS[-1], label=run_name)
    
        pareto_pts = is_pareto_efficient(np.array([score_list, sa_list]).T)
        plt.plot(pareto_pts[:, 0], pareto_pts[:, 1], marker, label=run_name)
    
    plt.legend(numpoints=1)
    plt.savefig("./tmp.png")