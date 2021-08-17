from pathlib import Path
import os
from rdkit import Chem, RDLogger

TASK_DIR = "../resource/task/"

RDLogger.logger().setLevel(RDLogger.CRITICAL)

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol) 
    except:
        return None
    
    if len(smiles) == 0:
        return None
        
    return smiles

def is_valid_smiles(smiles):
    mol = canonicalize(smiles)
    if mol is None or len(mol) == 0:
        return False

    return True
