from multiprocessing import Pool

from util.molecule.scoring.guacamol import (
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
from util.molecule.scoring.chemprop import jnk3_model, gsk3_model, qed_func, sa_func
from util.molecule.scoring.penalized_logp import penalized_logp_cyclebasis

from joblib import Parallel, delayed

from tqdm import tqdm

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

CHEMPROP_NAMES = [
    "jnk3",
    "gsk3",
    "qed",
    "sa",
]


def get_scoring_func(name):
    if name in GUACAMOL_NAMES:
        scoring_func = get_guacamol_scoring_func(name)
    elif name in CHEMPROP_NAMES:
        scoring_func = get_chemprop_scoring_func(name)

    return scoring_func


def get_guacamol_scoring_func(name):
    if name == "celecoxib":
        benchmark = similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )

    elif name == "troglitazone":
        benchmark = similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )

    elif name == "thiothixene":
        benchmark = similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )

    elif name == "aripiprazole":
        benchmark = similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        )

    elif name == "albuterol":
        benchmark = similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
            name="Albuterol",
            fp_type="FCFP4",
            threshold=0.75,
        )

    elif name == "mestranol":
        benchmark = similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        )

    elif name == "c11h24":
        benchmark = isomers_c11h24()

    elif name == "c9h10n2o2pf2cl":
        benchmark = isomers_c9h10n2o2pf2cl()

    elif name == "camphor_menthol":
        benchmark = median_camphor_menthol()

    elif name == "tadalafil_sildenafil":
        benchmark = median_tadalafil_sildenafil()

    elif name == "osimertinib":
        benchmark = hard_osimertinib()

    elif name == "fexofenadine":
        benchmark = hard_fexofenadine()

    elif name == "ranolazine":
        benchmark = ranolazine_mpo()

    elif name == "perindopril":
        benchmark = perindopril_rings()

    elif name == "amlodipine":
        benchmark = amlodipine_rings()

    elif name == "sitagliptin":
        benchmark = sitagliptin_replacement()

    elif name == "zaleplon":
        benchmark = zaleplon_with_other_formula()

    elif name == "valsartan_smarts":
        benchmark = valsartan_smarts()

    elif name == "decoration_hop":
        benchmark = decoration_hop()

    elif name == "scaffold_hop":
        benchmark = scaffold_hop()

    elif name == "penalized_logp":
        benchmark = penalized_logp_cyclebasis()

    elem_scoring_func = benchmark.wrapped_objective.score

    def scoring_func(smiles_list, jobs=8):
        if jobs == 0:
            scores = []
            for smiles in smiles_list:
                scores.append(elem_scoring_func(smiles))

        else:
            scores = Parallel(jobs)(
                delayed(elem_scoring_func)(smiles) for smiles in smiles_list
            )

        return scores

    return scoring_func


def get_chemprop_scoring_func(name):
    if name == "jnk3":
        batch_scoring_func = jnk3_model()
    elif name == "gsk3":
        batch_scoring_func = gsk3_model()
    elif name == "qed":
        batch_scoring_func = qed_func()
    elif name == "sa":
        batch_scoring_func = sa_func()

    def scoring_func(smiles_list, batch_size=10000):
        scores = []
        for start_idx in range(0, len(smiles_list), batch_size):
            end_idx = min(start_idx + batch_size, len(smiles_list))
            batch_scores = batch_scoring_func(smiles_list[start_idx:end_idx])
            scores.extend(batch_scores)

        return scores

    return scoring_func
