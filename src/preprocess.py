import torch
import json
import random

from data.util import load_smiles_list, load_tokenizer, create_vocabulary, load_vocabulary
from data.score import BindingScorer, PLogPScorer

if __name__ == "__main__":
    # create vocabulary
    smiles_list = []
    for split in ["train", "valid", "test"]:
        smiles_list += load_smiles_list("zinc", split)
    
    for split in ["train", "valid", "test"]:
        smiles_list += load_smiles_list("chembl24", split)
        
    for score_name in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        smiles_list += load_smiles_list(score_name, "default")
        
    tokenizer = load_tokenizer()
    vocabulary = create_vocabulary(smiles_list, tokenizer)
    print(vocabulary._tokens)
    torch.save(vocabulary._tokens, '../resource/data/vocab.pth')
    
    vocabulary = load_vocabulary()
    print(vocabulary._tokens)

    # create dataset for plogp
    smiles_list = load_smiles_list("zinc", "train")
    random.shuffle(smiles_list)
    smiles_list = smiles_list[:5000]
            
    with open("../resource/data/plogp/raw/zinc/plogp_smiles.txt", "w") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")
    
    scorer = PLogPScorer()
    score_list = scorer(smiles_list)
    with open("../resource/data/plogp/raw/zinc/plogp_score.txt", "w") as f:
        for score in score_list:
            f.write(str(score) + "\n")

