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
from model.ae import RnnAutoEncoder
from scoring.factory import get_scoring_func
from util.mol import randomize_smiles
from util.priority_queue import MaxRewardPriorityQueue

from tqdm import tqdm
import neptune.new as neptune

def train(model, optimizer, storage, vocab, tokenizer, scoring_func, num_samples, num_updates, sample_size, batch_size, warmup):
    for _ in range(num_samples):
        with torch.no_grad():
            seqs, lengths, _ = model.global_sample(sample_size, vocab)
        
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
    statistics["score/guacamol"] = (torch.topk(scores, k=100)[0].mean().item() + torch.topk(scores, k=10)[0].mean().item() + scores.max().item()) / 3
    
    if not warmup:
        for _ in range(num_updates):
            strings, seqs, lengths, scores = storage.sample_batch(batch_size)

            seqs = [smiles2seq(string, tokenizer, vocab) for string in strings]
            lengths = torch.tensor([seq.size(0) for seq in seqs])
            seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

            seqs = seqs.cuda()
            lengths = lengths.cuda()
            loss, step_statistics = model.global_step(seqs, lengths)
            for key, val in step_statistics.items():
                statistics[key] += val / num_updates
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, default="../resource/data/")
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/")
    parser.add_argument("--data_tag", type=str, default="zinc")
    parser.add_argument("--checkpoint_tag", type=str, default="default")

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--goal_dim", type=int, default=32)
    parser.add_argument("--load_path", default="")
    
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--goal_lr", type=float, default=1e-3)
    
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=1024)

    parser.add_argument("--num_updates", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--freeze_lstm", action="store_true")
    
    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(f"{args.data_dir}/{args.data_tag}.txt", "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer, args.max_length)
    dataset = SmilesDataset(smiles_list, tokenizer, vocab, transform=randomize_smiles)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )
    
    encoder = RnnEncoder(len(vocab), args.goal_dim, args.hidden_dim, args.num_layers)
    decoder = RnnDecoder(len(vocab), len(vocab), args.hidden_dim, args.goal_dim, args.num_layers)
    global_goal = torch.nn.Parameter(torch.randn(args.goal_dim))
    model = RnnAutoEncoder(encoder=encoder, decoder=decoder, global_goal=global_goal).cuda()
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    
    for idx, scoring_func_name in [
        (1, "celecoxib"), 
        (2, "troglitazone"), 
        (3, "thiothixene"), 
        (4, "aripiprazole"), 
        (5, "albuterol"), 
        (6, "mestranol"), 
        (7, "c11h24"), 
        (8, "c9h10n2o2pf2cl"), 
        (9, "camphor_menthol"), 
        (10, "tadalafil_sildenafil"), 
        (11, "osimertinib"), 
        (12, "fexofenadine"), 
        (13, "ranolazine"), 
        (14, "perindopril"), 
        (15, "amlodipine"), 
        (16, "sitagliptin"), 
        (17, "zaleplon"), 
        (18, "valsartan_smarts"), 
        (19, "decoration_hop"), 
        (20, "scaffold_hop"),
        (21, "penalized_logp"), 
        ]:
        
        model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_tag}.pth"))
        for param in model.encoder.parameters():
            param.requires_grad = False
        if args.freeze_decoder:
            for param in model.decoder.parameters():
                param.requires_grad = False
        if args.freeze_lstm:
            for param in model.decoder.lstm.parameters():
                param.requires_grad = False
            for param in model.decoder.encoder.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam([
            {"params": model.decoder.parameters()},
            {"params": [global_goal], "lr": args.goal_lr}], 
            lr=1e-3
            )
        scoring_func = get_scoring_func(scoring_func_name)
        storage = MaxRewardPriorityQueue()

        warmup = True
        for step in tqdm(range(args.steps)):
            if step > args.warmup_steps:
                warmup = False

            statistics = train(model, optimizer, storage, vocab, tokenizer, scoring_func, args.num_samples, args.num_updates, args.sample_size, args.batch_size, warmup)
            print(statistics)

            for key, val in statistics.items():
                run[f"{idx}_{scoring_func_name}/{key}"].log(val)
    

