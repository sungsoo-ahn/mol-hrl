import argparse
import os
import random
from collections import defaultdict

import torch
from torch.optim import Adam
import torch.nn.functional as F

from vocabulary import SmilesTokenizer, create_vocabulary
from dataset import SmilesDataset
from model.rnn import RnnGenerator
from model.gnn import GnnProjector
from util.mutate import mutate
from util.mol import randomize_smiles

from tqdm import tqdm
import neptune.new as neptune

def compute_accuracy(logits, y, batch_size):
    y_pred = torch.argmax(logits, dim=-1)
    correct = (y_pred == y)
    correct[y == 0] = True
    elem_acc = correct[y != 0].float().mean()
    seq_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, seq_acc

def train(projector, generator, optimizer, loader, loss2_coef):
    statistics = defaultdict(float)
    for x_seq, x_aug_seq, lengths, aug_lengths, x_graph, x_aug_graph in tqdm(loader):        
        x_seq = x_seq.cuda()
        x_aug_seq = x_aug_seq.cuda()
        x_graph = x_graph.cuda()
        x_aug_graph = x_aug_graph.cuda()

        c = projector(x_graph)
        c_aug = projector(x_aug_graph)

        out, _ = generator(x_seq[:, :-1], h=None, c=c, lengths=lengths-1)
        out_aug, _ = generator(x_aug_seq[:, :-1], h=None, c=c_aug, lengths=aug_lengths-1)

        logits = out.view(-1, out.size(-1))
        logits_aug = out_aug.view(-1, out_aug.size(-1))

        y = x_seq[:, 1:].reshape(-1)
        y_aug = x_aug_seq[:, 1:].reshape(-1)
        
        loss0 = F.cross_entropy(logits, y, reduction="sum", ignore_index=0)
        loss0 /= torch.sum(lengths - 1)

        loss1 = F.cross_entropy(logits_aug, y_aug, reduction="sum", ignore_index=0)
        loss1 /= torch.sum(aug_lengths - 1)

        loss2 = ((c - c_aug) ** 2).mean().sqrt()

        loss = loss0 + loss1 + loss2_coef * loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elem_acc, seq_acc = compute_accuracy(logits, y, x_seq.size(0))
        elem_acc_aug, seq_acc_aug = compute_accuracy(logits_aug, y_aug, x_aug_seq.size(0))

        statistics["loss"] += loss.item()
        statistics["loss0"] += loss0.item()
        statistics["loss1"] += loss1.item()
        statistics["loss2"] += loss2.item()
        statistics["elem_acc"] += elem_acc.item()
        statistics["seq_acc"] += seq_acc.item()
        statistics["elem_acc_aug"] += elem_acc_aug.item()
        statistics["seq_acc_aug"] += seq_acc_aug.item()

    for key in statistics:
        statistics[key] /= len(loader)
    
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/full.txt")
    
    parser.add_argument("--generator_hidden_dim", type=int, default=1024)
    parser.add_argument("--generator_num_layers", type=int, default=3)

    parser.add_argument("--projector_hidden_dim", type=int, default=32)
    parser.add_argument("--projector_num_layers", type=int, default=5)
    parser.add_argument("--proj_dim", type=int, default=256)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--save_dir", default="./checkpoint/zinc/")

    parser.add_argument("--loss2_coef", type=float, default=1e-1)
    parser.add_argument("--randomize_smiles", action="store_true")

    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer)
    if args.randomize_smiles:
        transform = randomize_smiles
    else:
        transform = None

    dataset = SmilesDataset(smiles_list, tokenizer, vocab, return_seq=True, return_graph=True, transform=transform, aug_transform=mutate)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    generator = RnnGenerator(len(vocab), len(vocab), args.generator_hidden_dim, args.proj_dim, args.generator_num_layers).cuda()
    projector = GnnProjector(args.projector_num_layers, args.projector_hidden_dim, args.proj_dim).cuda()
    optimizer = torch.optim.Adam(list(generator.parameters()) + list(projector.parameters()), lr=1e-3)
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    neptune_run_id = run["sys/id"].fetch()
    checkpoint_dir = f"./checkpoint/{neptune_run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        statistics = train(projector, generator, optimizer, loader, args.loss2_coef)
        for key in statistics:
            run[key].log(statistics[key])
        
        state_dict = {
            "projector": projector.state_dict(),
            "generator": generator.state_dict()
        }
        torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")