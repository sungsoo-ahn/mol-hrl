import argparse
import os
import random
from collections import defaultdict
from functools import total_ordering

import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F

from vocabulary import SmilesTokenizer, create_vocabulary
from dataset import SmilesDataset
from model.rnn import RnnGenerator
from scoring.factory import get_scoring_func


from tqdm import tqdm
import neptune.new as neptune

def train(generator, code, optimizer, scoring_func, vocab, tokenizer, batch_size, max_length):
    statistics = defaultdict(float)
    code_ = code.unsqueeze(0).expand(batch_size, code.size(0))
    strings, log_probs = generator.sample_strings(code_, batch_size, max_length, vocab, tokenizer)
    scores = scoring_func(strings)
    scores = torch.tensor(scores).cuda()
    
    topk_idxs = torch.topk(scores, 8)[1]
    loss = -torch.mean(log_probs[topk_idxs])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    statistics["loss"] = loss.item()
    statistics["score/max"] = scores.max().item()
    statistics["score/mean"] = scores.mean().item()

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/test.txt")
    
    parser.add_argument("--generator_hidden_dim", type=int, default=1024)
    parser.add_argument("--generator_num_layers", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--load_path", default="")
    
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=81)
    parser.add_argument("--batch_size", type=int, default=1024)
    
    
    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer)
    
    generator = RnnGenerator(len(vocab), len(vocab), args.generator_hidden_dim, args.proj_dim, args.generator_num_layers).cuda()
    
    state_dict = torch.load(args.load_path)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False


    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)

    for scoring_func_name in [
        "penalized_logp", 
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
        "scaffold_hop"
        ]:
        code = torch.nn.Parameter(torch.randn(args.proj_dim).cuda())
        optimizer = torch.optim.Adam([code], lr=1e-1)
        scoring_func = get_scoring_func(scoring_func_name)

        for step in tqdm(range(args.steps)):
            statistics = train(generator, code, optimizer, scoring_func, vocab, tokenizer, args.batch_size, args.max_length)
            for key in statistics:
                run[f"{scoring_func_name}/{key}"].log(statistics[key])