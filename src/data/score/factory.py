from multiprocessing import Pool

from data.score.guacamol import (
    similarity,
    isomers_c11h24,
    isomers_c9h10n2o2pf2cl,
    median_camphor_menthol,
    median_tadalafil_sildenafil,
    hard_osimertinib,
    hard_fexofenadine,
    ranolazine_mpo,
    perindopril_rings,
    amlodipine_rings,
    sitagliptin_replacement,
    zaleplon_with_other_formula,
    valsartan_smarts,
    decoration_hop,
    scaffold_hop,
)
from data.score.penalized_logp import penalized_logp_cyclebasis
from guacamol.common_scoring_functions import RdkitScoringFunction
from guacamol.utils.descriptors import logP, qed, tpsa, mol_weight

from joblib import Parallel, delayed

GUACAMOL_NAMES = [
    "celecoxib",
    "troglitazone",
    "thiothixene",
    "aripiprazole",
    "albuterol",
    "mestranol",
    "c11h24",
    "c9h10n2o2pf2cl",
    "camphor_menthol",
    "tadalafil_sildenafil",
    "osimertinib",
    "fexofenadine",
    "ranolazine",
    "perindopril",
    "amlodipine",
    "sitagliptin",
    "zaleplon",
    "valsartan_smarts",
    "decoration_hop",
    "scaffold_hop",
    "penalized_logp",
]


def get_scoring_func(name, num_workers=32):
    if name == "celecoxib":
        benchmark = similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
        objective = benchmark.objective

    elif name == "troglitazone":
        benchmark = similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
        objective = benchmark.objective

    elif name == "thiothixene":
        benchmark = similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
        objective = benchmark.objective

    elif name == "aripiprazole":
        benchmark = similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        )
        objective = benchmark.objective

    elif name == "albuterol":
        benchmark = similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", name="Albuterol", fp_type="FCFP4", threshold=0.75,
        )
        objective = benchmark.objective

    elif name == "mestranol":
        benchmark = similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        )
        objective = benchmark.objective

    elif name == "c11h24":
        benchmark = isomers_c11h24()
        objective = benchmark.objective

    elif name == "c9h10n2o2pf2cl":
        benchmark = isomers_c9h10n2o2pf2cl()
        objective = benchmark.objective

    elif name == "camphor_menthol":
        benchmark = median_camphor_menthol()
        objective = benchmark.objective

    elif name == "tadalafil_sildenafil":
        benchmark = median_tadalafil_sildenafil()
        objective = benchmark.objective

    elif name == "osimertinib":
        benchmark = hard_osimertinib()
        objective = benchmark.objective

    elif name == "fexofenadine":
        benchmark = hard_fexofenadine()
        objective = benchmark.objective

    elif name == "ranolazine":
        benchmark = ranolazine_mpo()
        objective = benchmark.objective

    elif name == "perindopril":
        benchmark = perindopril_rings()
        objective = benchmark.objective

    elif name == "amlodipine":
        benchmark = amlodipine_rings()
        objective = benchmark.objective

    elif name == "sitagliptin":
        benchmark = sitagliptin_replacement()
        objective = benchmark.objective

    elif name == "zaleplon":
        benchmark = zaleplon_with_other_formula()
        objective = benchmark.objective

    elif name == "valsartan_smarts":
        benchmark = valsartan_smarts()
        objective = benchmark.objective

    elif name == "decoration_hop":
        benchmark = decoration_hop()
        objective = benchmark.objective

    elif name == "scaffold_hop":
        benchmark = scaffold_hop()
        objective = benchmark.objective

    elif name == "penalized_logp":
        benchmark = penalized_logp_cyclebasis()
        objective = benchmark.objective

    elif name == "molwt":
        objective = RdkitScoringFunction(descriptor=mol_weight)
        objective.corrupt_score = -1e6

    elif name == "logp":
        objective = RdkitScoringFunction(descriptor=logP)
        objective.corrupt_score = -1e6

    elif name == "qed":
        objective = RdkitScoringFunction(descriptor=qed)
        objective.corrupt_score = -1e6

    elif name == "tpsa":
        objective = RdkitScoringFunction(descriptor=tpsa)
        objective.corrupt_score = -1e6

    # print(objective.corrupt_score)

    float_score = lambda smiles: float(objective.score(smiles))

    def parallel_scoring_func(smiles_list):
        scores = Parallel(num_workers)(delayed(float_score)(smiles) for smiles in smiles_list)

        return scores

    return objective.score, parallel_scoring_func, objective.corrupt_score
