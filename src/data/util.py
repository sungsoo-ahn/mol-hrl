import os
from pathlib import Path

from tokenizers import Tokenizer

from docking_benchmark.data.proteins import get_proteins

TASK_DIR = "../resource/data/"

def load_tokenizer():
    tokenizer = Tokenizer.from_file(f"{TASK_DIR}/tokenizer.json")
    return tokenizer

def load_smiles_list(task, split):
    if task == "zinc":
        smiles_list_path = os.path.join(TASK_DIR, f"plogp/raw/zinc/{split}.txt")
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        return smiles_list

    elif task == "chembl24":
        smiles_list_path = os.path.join(TASK_DIR, f"docking/raw/data/datasets/chembl24/{split}.smiles")
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        return smiles_list
    
    elif task in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        protein = get_proteins()[task]
        smiles_list = protein.datasets[split][0]
        return smiles_list
    
    elif task == "plogp":
        smiles_list_path = os.path.join(TASK_DIR, f"plogp/raw/zinc/plogp_smiles.txt")
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        return smiles_list
    
def load_score_list(task, split):
    if task in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        protein = get_proteins()[task]
        score_list = protein.datasets[split][1]
        return score_list
    
    elif task == "plogp":
        score_list_path = os.path.join(TASK_DIR, f"plogp/raw/zinc/plogp_score.txt")
        score_list = list(map(float, Path(score_list_path).read_text(encoding="utf-8").splitlines()))
        return score_list