import argparse
from tqdm import tqdm
from rdkit import Chem
from mol_editor.scoring.factory import get_scoring_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="raw molecule dataset, one molecule per line")
    parser.add_argument("--cooked", required=True, help="cooked score data, one score per line")
    parser.add_argument("--scoring", required=True, help="name of scoring function to use")
    args = parser.parse_args()
       
    smiles_list = []
    with open(args.raw, "r") as f:
        for line in tqdm(f):
            smi = line.strip("\r\n ").split(",")[0]
            smiles_list.append(smi)
            
    scoring_func = get_scoring_func(args.scoring)
        
    score_list = scoring_func(smiles_list)
        
    with open(args.cooked, "w") as f:
        f.write("\n".join(map(str, score_list)))
