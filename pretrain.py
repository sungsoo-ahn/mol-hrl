import os
import argparse
from collections import defaultdict

import torch
from torch.optim import Adam

from vocabulary import SmilesTokenizer, create_vocabulary
from dataset import SmilesDataset
from model.rnn import RnnDecoder, RnnEncoder
from model.ae import RnnAutoEncoder
from util.mol import randomize_smiles

from tqdm import tqdm
import neptune.new as neptune

def train(model, optimizer, loader):
    statistics = defaultdict(float)
    for seqs, lengths in tqdm(loader):        
        seqs = seqs.cuda()
        lengths = lengths.cuda()

        loss, step_statistics = model.step(seqs, lengths)
        for key, val in step_statistics.items():
            statistics[key] += val
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for key in statistics:
        statistics[key] /= len(loader)
    
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/full.txt")
    parser.add_argument("--max_length", type=int, default=100)
    
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--goal_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--tag", type=str, default="default")

    parser.add_argument("--randomize_smiles", action="store_true")

    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer, args.max_length)
    dataset = SmilesDataset(smiles_list, tokenizer, vocab, transform=randomize_smiles)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    encoder = RnnEncoder(len(vocab), args.goal_dim, args.hidden_dim, args.num_layers)
    decoder = RnnDecoder(len(vocab), len(vocab), args.hidden_dim, args.goal_dim, args.num_layers)
    global_goal = torch.nn.Parameter(torch.randn(args.goal_dim))
    model = RnnAutoEncoder(encoder=encoder, decoder=decoder, global_goal=global_goal).cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)

    best_train_acc = -10.0
    for epoch in range(args.epochs):
        statistics = train(model, optimizer, loader)
        print(statistics)
        for key in statistics:
            run[key].log(statistics[key])

        train_acc = statistics["acc/recon/seq"]
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            state_dict = model.state_dict()
            torch.save(state_dict, f"./checkpoint/{args.tag}.pth")
