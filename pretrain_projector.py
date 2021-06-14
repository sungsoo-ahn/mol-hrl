import argparse
import os
import random
from collections import defaultdict

import torch
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

from dataset import SmilesDataset
from model.gnn import GnnProjector
from util.mutate import mutate

from tqdm import tqdm
import neptune.new as neptune

def compute_accuracy(logits, labels):
    return (torch.argmax(logits, dim=-1) == labels).float().mean()

def train(projector, linear, optimizer, loader):
    statistics = defaultdict(float)
    for (batched_pyg_data, ) in tqdm(loader):
        batched_pyg_data = batched_pyg_data.cuda()

        node_h = projector(batched_pyg_data)
        graph_h = global_add_pool(node_h, batched_pyg_data.batch)
        graph_h = linear(graph_h)
        
        node_h = F.normalize(node_h, p=2, dim=1)
        graph_h = F.normalize(graph_h, p=2, dim=1)

        logits = torch.matmul(node_h, graph_h.T)
        labels = torch.arange(node_h.size(0))[batched_pyg_data.batch].cuda()
        
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(logits, labels)
        
        statistics["loss"] += loss.item()
        statistics["acc"] += acc.item()
        
    for key in statistics:
        statistics[key] /= len(loader)
    
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/full.txt")
    
    parser.add_argument("--projector_hidden_dim", type=int, default=256)
    parser.add_argument("--projector_num_layers", type=int, default=5)
    parser.add_argument("--code_dim", type=int, default=32)
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--save_dir", default="./checkpoint/zinc/")
    
    args = parser.parse_args()

    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()
    
    dataset = SmilesDataset(smiles_list, None, None, return_seq=False, return_graph=True, transform=None)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=16,
    )

    projector = GnnProjector(args.projector_num_layers, args.projector_hidden_dim, args.code_dim).cuda()
    linear = torch.nn.Linear(args.code_dim, args.code_dim).cuda()
    optimizer = Adam(list(projector.parameters()) + list(linear.parameters()), lr=1e-3)
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    neptune_run_id = run["sys/id"].fetch()
    checkpoint_dir = f"./checkpoint/{neptune_run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        statistics = train(projector, linear, optimizer, loader)
        print(statistics)
        for key in statistics:
            run[key].log(statistics[key])
        
        state_dict = {"projector": projector.state_dict()}
        torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")