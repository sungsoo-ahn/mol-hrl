import argparse
import os
import random
from collections import defaultdict

import torch
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

from vocabulary import SmilesTokenizer, create_vocabulary
from dataset import SmilesDataset
from model.rnn import RnnGenerator
from model.gnn import GnnProjector
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

def train(projector, generator, optimizer, loader):
    statistics = defaultdict(float)
    for x_seq, lengths, batched_pyg_data in tqdm(loader):        
        x_seq = x_seq.cuda()
        batched_pyg_data = batched_pyg_data.cuda()
        
        with torch.no_grad():
            node_h = projector(batched_pyg_data)
            code = global_add_pool(node_h, batched_pyg_data.batch)

        out, _ = generator(x_seq[:, :-1], h=None, c=code, lengths=lengths-1)
        logits = out.view(-1, out.size(-1))        
        y = x_seq[:, 1:].reshape(-1)

        loss = F.cross_entropy(logits, y, reduction="sum", ignore_index=0)
        loss /= torch.sum(lengths - 1)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        elem_acc, seq_acc = compute_accuracy(logits, y, x_seq.size(0))

        statistics["loss"] += loss.item()
        statistics["elem_acc"] += elem_acc.item()
        statistics["seq_acc"] += seq_acc.item()

    for key in statistics:
        statistics[key] /= len(loader)
    
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/full.txt")
    
    parser.add_argument("--code_dim", type=int, default=32)
    
    parser.add_argument("--generator_hidden_dim", type=int, default=1024)
    parser.add_argument("--generator_num_layers", type=int, default=3)

    parser.add_argument("--projector_hidden_dim", type=int, default=256)
    parser.add_argument("--projector_num_layers", type=int, default=5)
    parser.add_argument("--projector_load_path", type=str)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--save_dir", default="./checkpoint/zinc/")

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

    dataset = SmilesDataset(smiles_list, tokenizer, vocab, return_seq=True, return_graph=True, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    generator = RnnGenerator(len(vocab), len(vocab), args.generator_hidden_dim, args.code_dim, args.generator_num_layers).cuda()
    projector = GnnProjector(args.projector_num_layers, args.projector_hidden_dim, args.code_dim).cuda()
    projector.load_state_dict(torch.load(args.projector_load_path)["projector"])
    projector.eval()
    optimizer = Adam(generator.parameters(), lr=1e-3)
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    neptune_run_id = run["sys/id"].fetch()
    checkpoint_dir = f"./checkpoint/{neptune_run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        statistics = train(projector, generator, optimizer, loader)
        print(statistics)
        for key in statistics:
            run[key].log(statistics[key])

        state_dict = {
            "projector": projector.state_dict(),
            "generator": generator.state_dict()
        }
        torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")