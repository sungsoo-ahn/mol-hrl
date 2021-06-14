import argparse
import os
from collections import defaultdict
from functools import total_ordering

import torch
import torch.nn.functional as F

from vocabulary import SmilesTokenizer, create_vocabulary
from model.rnn import RnnGenerator
from model.gnn import GnnProjector
from pretrain_vae import VAE
from scoring.factory import get_scoring_func


from tqdm import tqdm
import neptune.new as neptune

from torch.distributions import Normal

class HigherModel(torch.nn.Module):
    def __init__(self, proj_dim):
        super(HigherModel, self).__init__()
        self.mean = torch.nn.Parameter(torch.zeros(proj_dim))
        self.var = torch.nn.Parameter(torch.zeros(proj_dim))
    
    def sample(self, batch_size):
        mean_ = self.mean.unsqueeze(0).expand(batch_size, self.mean.size(0))
        std_ = torch.exp(self.var.unsqueeze(0).expand(batch_size, self.var.size(0)) / 2)
        distribution = Normal(mean_, std_)
        code = distribution.sample()
        log_probs = distribution.log_prob(code)

        return code, log_probs

def train(model, model_optimizer, higher_model, higher_optimizer, scoring_func, vocab, tokenizer, batch_size, max_length):
    statistics = defaultdict(float)
    code, log_probs = higher_model.sample(batch_size)
    
    with torch.no_grad():
        strings = model.decoder.max_sample_strings(code, batch_size, max_length, vocab, tokenizer)
    
    scores = scoring_func(strings)
    scores = torch.tensor(scores).cuda()
    
    topk_idxs = torch.topk(scores, 1)[1]
    loss = -torch.mean(log_probs[topk_idxs])
    
    higher_optimizer.zero_grad()
    loss.backward()
    higher_optimizer.step()

    topk_strings = [strings[idx] for idx in topk_idxs.tolist()]
    topk_pyg_


    statistics["loss"] = loss.item()
    statistics["score/max"] = scores.max().item()
    statistics["score/mean"] = scores.mean().item()
    statistics["ratio"] = len(set(strings)) / len(strings)

    print(statistics)

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
    
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=81)
    parser.add_argument("--batch_size", type=int, default=1024)
    
    
    args = parser.parse_args()

    tokenizer = SmilesTokenizer()
    with open(args.smiles_list_path, "r") as f:
        smiles_list = f.read().splitlines()

    vocab = create_vocabulary(smiles_list, tokenizer)
    
    generator = RnnGenerator(len(vocab), len(vocab), args.generator_hidden_dim, args.code_dim, args.generator_num_layers)
    projector = GnnProjector(args.projector_num_layers, args.projector_hidden_dim, args.code_dim)
    for param in projector.parameters():
        param.requires_grad = False
    
    model = VAE(encoder=projector, decoder=generator, code_dim=args.code_dim).cuda()
    state_dict = torch.load(args.load_path)
    model.encoder.load_state_dict(state_dict["generator"])
    model.decoder.load_state_dict(state_dict["projector"])
    model.fc_mu.load_state_dict(state_dict["fc_mu"])
    model.fc_var.load_state_dict(state_dict["fc_var"])
    
    model_optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-3)

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
        higher_model = HigherModel(args.proj_dim).cuda()
        higher_optimizer = torch.optim.Adam(higher_model.parameters(), lr=1e-1)

        scoring_func = get_scoring_func(scoring_func_name)
        for step in tqdm(range(args.steps)):
            statistics = train(model, model_optimizer, higher_model, higher_optimizer, scoring_func, vocab, tokenizer, args.batch_size, args.max_length)
            for key in statistics:
                run[f"{scoring_func_name}/{key}"].log(statistics[key])