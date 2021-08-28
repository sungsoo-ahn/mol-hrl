from tqdm import tqdm

import os, sys
import networkx as nx
from joblib import Parallel, delayed

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from docking_benchmark.data.proteins import get_proteins
from data.smiles.util import canonicalize

from moses.metrics import internal_diversity

import torch

def _raw_plogp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None
    
    LOGP_MEAN = 2.4570965532649507
    LOGP_STD = 1.4339810636722639
    SASCORE_MEAN = 3.0508333383104556
    SASCORE_STD = 0.8327034846660627
    CYCLESCORE_MEAN = 0.048152237188108474
    CYCLESCORE_STD = 0.2860582871837183
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        log_p = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
    except:
        return None
        
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - CYCLESCORE_MEAN) / CYCLESCORE_STD

    return log_p - sa_score - cycle_score
    
def _raw_binding_scorer(smiles, protein, key):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None

    protein = get_proteins()[protein]
    return protein.dock_smiles_to_protein(smiles)[key]

class PLogPScorer(object):
    def __init__(self, num_workers=8):
        self.pool = Parallel(n_jobs=num_workers) if num_workers > 0 else None
        self._score_smiles = _raw_plogp
        self.success_margin = 0.5

    def __call__(self, smiles_list, query=None, success_margin=0.5):
        statistics = dict()
        unique_smiles_list = list(set(smiles_list))
        statistics["unique_ratio"] = float(len(set(unique_smiles_list))) / len(smiles_list)
        
        score_list = self.pool(delayed(self._score_smiles)(smiles) for smiles in unique_smiles_list)
        unique_pairs = zip(unique_smiles_list, score_list)
        valid_pairs = [(smiles, score) for smiles, score in unique_pairs if score is not None]
        statistics["unique_valid_ratio"] = float(len(valid_pairs)) / len(smiles_list)

        if query is not None:
            if len(valid_pairs) > 0:
                valid_scores_tsr = torch.FloatTensor([score for _, score in valid_pairs])
                statistics["mae_score"] = (query - valid_scores_tsr).abs().mean().item()
                statistics["mean_score"] = valid_scores_tsr.mean().item()
                statistics["std_score"] = valid_scores_tsr.std().item() if len(valid_pairs) > 1 else 0.0
                statistics["max_score"] = valid_scores_tsr.max().item()

                def is_success(score):
                    return (score > query - self.success_margin) and (score < query + self.success_margin)

                success_pairs = [(smiles, score) for smiles, score in valid_pairs if is_success(score)]
                statistics["success_ratio"] = float(len(success_pairs)) / len(smiles_list)
                
                if len(success_pairs) > 0:
                    success_mols = [Chem.MolFromSmiles(smiles) for smiles, _ in success_pairs]
                    statistics["internal_diversity"] = internal_diversity(success_mols)

            else:
                statistics["success_ratio"] = 0.0

        return statistics
        
        
class BindingScorer(PLogPScorer):
    def __init__(self, protein, key, num_workers=8):
        self.pool = Parallel(n_jobs=num_workers) if num_workers > 0 else None
        self._score_smiles = lambda smiles: _raw_binding_scorer(smiles, protein, key)

    
def load_scorer(task):
    if task == "plogp":
        scorer = PLogPScorer() 
    elif task in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        scorer = BindingScorer(task, "default")

    return scorer
