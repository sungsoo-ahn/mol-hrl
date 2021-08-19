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

    mol = Chem.MolFromSmiles(smiles)
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

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

    def __call__(self, smiles_or_smiles_list):
        if isinstance(smiles_or_smiles_list, list):
            if self.pool is None:
                score_list = [self._score_smiles(smiles) for smiles in smiles_or_smiles_list]
            else:
                score_list = self.pool(delayed(self._score_smiles)(smiles) for smiles in smiles_or_smiles_list)
            return score_list
        
        elif isinstance(smiles_or_smiles_list, str):
            return self._score_smiles(smiles_or_smiles_list)
        
        else:
            print(smiles_or_smiles_list)
            assert False
    
        
        
class BindingScorer(PLogPScorer):
    def __init__(self, protein, key, num_workers=8):
        self.pool = Parallel(n_jobs=num_workers) if num_workers > 0 else None
        self._score_smiles = lambda smiles: _raw_binding_scorer(smiles, protein, key)