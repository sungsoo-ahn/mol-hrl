import argparse
import os
from collections import defaultdict
from functools import total_ordering

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vocabulary import SmilesTokenizer, create_vocabulary, seq2smiles, smiles2seq
from dataset import SmilesDataset
from model.rnn import RnnDecoder, RnnEncoder
from model.vae import RnnVariationalAutoEncoder
from scoring.factory import get_scoring_func
from util.mol import randomize_smiles
from util.priority_queue import MaxRewardPriorityQueue

from tqdm import tqdm
import neptune.new as neptune

def train(model, optimizer, storage, vocab, tokenizer, scoring_func, sample_size, warmup):
    for _ in range(8):
        with torch.no_grad():
            seqs, lengths, _ = model.sample_seq(sample_size, vocab)
        
        strings = seq2smiles(seqs, tokenizer, vocab)
        scores = scoring_func(strings)    
        seqs = seqs.cpu().split(1, dim=0)
        lengths = lengths.cpu().split(1, dim=0)
        
        storage.add_list(smis=strings, seqs=seqs, lengths=lengths, scores=scores)
        storage.squeeze_by_kth(k=1024)
    
    strings, seqs, lengths, scores = storage.get_elems()
    scores = torch.tensor(scores)

    statistics = defaultdict(float)
    statistics["score/max"] = scores.max().item()
    statistics["score/mean"] = scores.mean().item()
    statistics["score/123"] = (torch.topk(scores, k=100)[0].mean().item() + torch.topk(scores, k=10)[0].mean().item() + scores.max().item()) / 3
    #statistics["ratio"] = len(set(strings)) / len(strings)

    if not warmup:
        seqs = [smiles2seq(string, tokenizer, vocab) for string in strings]
        lengths = torch.tensor([seq.size(0) for seq in seqs])
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

        seqs = seqs.cuda()
        lengths = lengths.cuda()
        loss, step_statistics = model.step(seqs, lengths)
        for key, val in step_statistics.items():
            statistics[key] += val
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
    
    parser.add_argument("--steps", type=int, default=200)
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
    model = RnnVariationalAutoEncoder(encoder=encoder, decoder=decoder, code_dim=args.code_dim).cuda()

    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    
    for scoring_func_name in [
        #"penalized_logp", 
        #"celecoxib", 
        #"troglitazone", 
        #"thiothixene", 
        #"aripiprazole", 
        #"albuterol", 
        #"mestranol", 
        #"c11h24", 
        #"c9h10n2o2pf2cl", 
        "camphor_menthol", 
        "tadalafil_sildenafil", 
        "osimertinib", 
        "fexofenadine", 
        "ranolazine", 
        "perindopril", 
        "amlodipine", 
        "sitagliptin", 
        "zaleplon", 
        #"valsartan_smarts", 
        #"decoration_hop", 
        #"scaffold_hop"
        ]:
        
        model.load_state_dict(torch.load(args.load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scoring_func = get_scoring_func(scoring_func_name)
        storage = MaxRewardPriorityQueue()

        warmup = True
        for step in tqdm(range(args.steps)):
            if step > 5:
                warmup = False

            statistics = train(model, optimizer, storage, vocab, tokenizer, scoring_func, args.sample_size, warmup)
            print(statistics)

            for key, val in statistics.items():
                run[f"{scoring_func_name}/{key}"].log(val)
    

