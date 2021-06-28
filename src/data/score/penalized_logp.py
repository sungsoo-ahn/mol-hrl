import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import os, sys

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

import networkx as nx


from guacamol.common_scoring_functions import (
    TanimotoScoringFunction,
    RdkitScoringFunction,
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    SMARTSScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    MoleculewiseScoringFunction,
)
from guacamol.utils.descriptors import (
    num_rotatable_bonds,
    num_aromatic_rings,
    logP,
    qed,
    tpsa,
    bertz,
    mol_weight,
    AtomCounter,
    num_rings,
)

LOGP_MEAN = 2.4570965532649507
LOGP_STD = 1.4339810636722639
SASCORE_MEAN = 3.0508333383104556
SASCORE_STD = 0.8327034846660627
CYCLESCORE_MEAN = 0.048152237188108474
CYCLESCORE_STD = 0.2860582871837183


def _penalized_logp_cyclebasis(mol: Mol):
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - CYCLESCORE_MEAN) / CYCLESCORE_STD

    return log_p - sa_score - cycle_score


def penalized_logp_cyclebasis():
    benchmark_name = "Penalized logP CycleBasis"
    objective = RdkitScoringFunction(descriptor=lambda mol: _penalized_logp_cyclebasis(mol))
    objective.corrupt_score = -1000.0
    specification = uniform_specification(1)
    return GoalDirectedBenchmark(
        name=benchmark_name, objective=objective, contribution_specification=specification,
    )
