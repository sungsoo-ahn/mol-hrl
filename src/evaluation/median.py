from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

from data.smiles.util import load_smiles_list

def get_fp(mol):
    return FingerprintMols.FingerprintMol(
        mol,
        minPath=1,
        maxPath=7,
        fpSize=2048,
        bitsPerHash=2,
        useHs=True,
        tgtDensity=0.0,
        minSize=128,
    )
        

def run_median(model, run):
    smiles_list = load_smiles_list(model.hparams.data_dir, split="train_labeled")
    smiles_list0 = smiles_list[:1024]
    smiles_list1 = smiles_list[1024:2048]
    
    mol0s = [Chem.MolFromSmiles(smiles) for smiles in smiles_list0]
    mol1s = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1]
    
    fp0s = [get_fp(mol) for mol in mol0s]
    fp1s = [get_fp(mol) for mol in mol1s]
    
    model.eval()
    out = model.encoder.encode_smiles(smiles_list0, model.device)
    _, _, codes0 = model.compute_codes(out)
    out = model.encoder.encode_smiles(smiles_list1, model.device)
    _, _, codes1 = model.compute_codes(out)
    
    for alpha in tqdm(np.linspace(0.0, 1.0, num=1000)):
        codes = alpha * codes0 + (1-alpha) * codes1
        if model.hparams.ae_type in ["sae", "cae"]:
            codes = F.normalize(codes, p=2, dim=1)

        smiles_list = model.decoder.decode_smiles(codes, deterministic=True)
        num_valids = 0
        avg_sim0 = 0.0
        avg_sim1 = 0.0
        avg_sim01 = 0.0
        for smiles, fp0, fp1 in zip(smiles_list, fp0s, fp1s):
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = get_fp(mol)
                num_valids += 1
            except:
                continue
            
            sim0 = DataStructs.TanimotoSimilarity(fp, fp0)
            sim1 = DataStructs.TanimotoSimilarity(fp, fp1)
            sim01 = (sim0 * sim1) ** 0.5 

            avg_sim0 += sim0 
            avg_sim1 += sim1
            avg_sim01 += sim01
        
        run["median/clean_ratio"].log(float(num_valids) / 1024)
        if num_valids > 0:
            run["median/sim0"].log(avg_sim0 / num_valids)
            run["median/sim1"].log(avg_sim1 / num_valids)
            run["median/sim01"].log(avg_sim01 / num_valids)
        else:
            run["median/sim0"].log(0.0)
            run["median/sim1"].log(0.0)
            run["median/sim01"].log(0.0)
