import argparse
import os
from collections import defaultdict
from functools import total_ordering

import torch
import torch.nn.functional as F

from vocabulary import SmilesTokenizer, create_vocabulary, seq2smiles
from dataset import SmilesDataset
from model.rnn import RnnDecoder, RnnEncoder
from model.ae import RnnAutoEncoder
from model.hagent import HierarchicalPolicy
from scoring.factory import get_scoring_func
from util.mol import randomize_smiles


from tqdm import tqdm
import neptune.new as neptune

from torch.distributions import Normal

class Policy(torch.nn.Module):
    def __init__(self, code_dim, decoder):
        super(Policy, self).__init__()
        self.code_mean = torch.nn.Parameter(torch.zeros(code_dim))
        self.code_logstd = torch.nn.Parameter(torch.zeros(code_dim))
        self.decoder = decoder
    
    def sample(self, sample_size, vocab):
        distribution = torch.distributions.Normal(self.code_mean, self.code_logstd.exp())
        codes = distribution.rsample([sample_size])
        seqs, lengths, log_probs = self.decoder.sample(codes, vocab, mode="sample")
        return seqs, lengths, log_probs

def train(policy, optimizer, vocab, tokenizer, scoring_func, sample_size):
    seqs, lengths, log_probs = policy.sample(sample_size, vocab)
    
    strings = seq2smiles(seqs, tokenizer, vocab)
    
    scores = scoring_func(strings)
    scores = torch.tensor(scores).cuda()
    
    topk_idxs = torch.topk(scores, 128)[1]
    loss = -torch.mean(log_probs[topk_idxs])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    statistics = dict()
    statistics["loss"] = loss.item()
    statistics["score/max"] = scores.max().item()
    statistics["score/mean"] = scores.mean().item()
    statistics["ratio"] = len(set(strings)) / len(strings)

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/test.txt")
    
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--code_dim", type=int, default=32)
    parser.add_argument("--load_path", default="")
    
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=81)
    parser.add_argument("--sample_size", type=int, default=1024)
    
    
    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer, args.max_length)
    dataset = SmilesDataset(smiles_list, tokenizer, vocab, transform=randomize_smiles)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )
    
    encoder = RnnEncoder(len(vocab), args.code_dim, args.hidden_dim, args.num_layers)
    decoder = RnnDecoder(len(vocab), len(vocab), args.hidden_dim, args.code_dim, args.num_layers)
    model = RnnAutoEncoder(encoder=encoder, decoder=decoder, code_dim=args.code_dim).cuda()

    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    
    for scoring_func_name in [
        #"penalized_logp", 
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
        
        model.load_state_dict(torch.load(args.load_path))
        policy = Policy(args.code_dim, decoder).cuda()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        scoring_func = get_scoring_func(scoring_func_name)

        for step in tqdm(range(args.steps)):
            statistics = train(policy, optimizer, vocab, tokenizer, scoring_func, args.sample_size)
            print(statistics)
            for key, val in statistics.items():
                run[f"{scoring_func_name}/{key}"].log(val)
    

