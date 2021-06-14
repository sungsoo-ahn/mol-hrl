import argparse
import os
import random
from collections import defaultdict

import torch
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

from vocabulary import SmilesTokenizer, create_vocabulary
from dataset import SmilesDataset
from model.rnn import RnnGenerator, compute_rnn_accuracy, compute_rnn_ce
from model.gnn import GnnProjector
from util.mol import randomize_smiles

from tqdm import tqdm
import neptune.new as neptune

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, code_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = torch.nn.Linear(code_dim, code_dim)
        self.fc_var = torch.nn.Linear(code_dim, code_dim)
        self.code_dim = code_dim
        self.kl_coef = 1e-1

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def _run_step(self, x_seq, lengths, batched_pyg_data):
        x = self.encoder(batched_pyg_data)
        x = global_mean_pool(x, batched_pyg_data.batch)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        logits, _ = self.decoder(x_seq[:, :-1], h=None, c=z, lengths=lengths-1)
        
        return z, logits, p, q

    def step(self, x_seq, lengths, batched_pyg_data, vocab, tokenizer):
        z, logits, p, q = self._run_step(x_seq, lengths, batched_pyg_data)
        y_seq = x_seq[:, 1:]

        recon_loss = compute_rnn_ce(logits, y_seq, lengths)
        recon_elem_acc, recon_acc = compute_rnn_accuracy(logits, y_seq, z.size(0))
        
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coef

        loss = kl + recon_loss

        with torch.no_grad():
            strings = self.decoder.max_sample_strings(z, z.size(0), 81, vocab, tokenizer)
        

        statistics = {
            "recon_loss": recon_loss.item(),
            "recon_elem_acc": recon_elem_acc.item(),
            "recon_acc": recon_acc.item(),
            "kl": kl.item(),
            "loss": loss.item(),
            "ratio": len(set(strings)) / len(strings),
        }
        return loss, statistics
    
    #def max_sample_strings(self, vocab, tokenizer):
    #    distribution = torch.distributions.Normal(torch.zeros(128, self.code_dim).cuda(), torch.ones(128, self.code_dim).cuda())
    #    code = distribution.sample()
    #    strings = generator.max_sample_strings(code, 128, 81, vocab, tokenizer)
    #    return strings


def train(model, optimizer, loader, vocab, tokenizer):
    statistics = defaultdict(float)
    for x_seq, lengths, batched_pyg_data in tqdm(loader):        
        x_seq = x_seq.cuda()
        batched_pyg_data = batched_pyg_data.cuda()

        loss, step_statistics = model.step(x_seq, lengths, batched_pyg_data, vocab, tokenizer)
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        optimizer.step()

        for key, val in step_statistics.items():
            statistics[key] += val
        
    for key in statistics:
        statistics[key] /= len(loader)
    
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smiles_list_path", type=str, default="./data/zinc/test.txt")
    
    parser.add_argument("--code_dim", type=int, default=256)
    
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
    dataset = SmilesDataset(smiles_list, tokenizer, vocab, return_seq=True, return_graph=True, transform=None)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    generator = RnnGenerator(len(vocab), len(vocab), args.generator_hidden_dim, args.code_dim, args.generator_num_layers)
    projector = GnnProjector(args.projector_num_layers, args.projector_hidden_dim, args.code_dim)
    model = VAE(encoder=projector, decoder=generator, code_dim=args.code_dim).cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    run = neptune.init(project='sungsahn0215/mol-hrl', source_files=["*.py", "**/*.py"])
    run["parameters"] = vars(args)
    neptune_run_id = run["sys/id"].fetch()
    checkpoint_dir = f"./checkpoint/{neptune_run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        statistics = train(model, optimizer, loader, vocab, tokenizer)
        print(statistics)
        for key in statistics:
            run[key].log(statistics[key])

        state_dict = {
            "projector": projector.state_dict(),
            "fc_mu": model.fc_mu.state_dict(), 
            "fc_var": model.fc_var.state_dict(),
            "generator": generator.state_dict(),
        }
        torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")
