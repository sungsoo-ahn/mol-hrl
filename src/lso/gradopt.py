from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset
from data.score.dataset import ScoreDataset
from data.score.factory import get_scoring_func

from lso.nn import train_nn
#from lso.gp import train_gp

def extract_batched_codes(model, batched_data, attack_steps=0, attack_epsilon=1e-2):
    statistics = dict()
    with torch.no_grad():
        batched_codes = model.ae.encode(batched_data)
    
    # Finetune codes
    batched_codes.requires_grad = True
    for _ in range(attack_steps):
        out = model.ae.decoder(batched_data, batched_codes)
        loss, statistics = model.ae.decoder.compute_recon_loss(out, batched_data)
        attack_grad = torch.autograd.grad(
            loss, batched_codes, retain_graph=False, create_graph=False
            )[0]
        
        batched_codes = batched_codes - attack_epsilon * attack_grad
        batched_codes = model.ae.project(batched_codes, batched_data)
    
    return batched_codes, statistics


def extract_codes(model, split):
    hparams = model.hparams
    if hparams.encoder_type == "seq":
        input_dataset_cls = SequenceDataset
    elif hparams.encoder_type == "graph":
        input_dataset_cls = GraphDataset

    dataset = input_dataset_cls(hparams.data_dir, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=hparams.num_workers,
    )

    codes = []
    for batched_data in tqdm(dataloader):
        if hparams.encoder_type == "seq":
            batched_data = tuple([item.cuda() for item in batched_data])
        elif hparams.encoder_type == "graph":
            batched_data = batched_data.cuda()
            
        batched_codes, _ = extract_batched_codes(model, batched_data)
        codes.append(batched_codes.detach().cpu())
    
    codes = torch.cat(codes, dim=0)
    return codes

def run_gradopt(model, regression_model_name, score_func_name, run, k=1024, steps=1000):
    # Prepare scoring function
    _, score_func, corrupt_score = get_scoring_func(score_func_name)
    
    # Prepare score dataset
    train_score_dataset = ScoreDataset(model.hparams.data_dir, [score_func_name], "train_labeled")
    val_score_dataset = ScoreDataset(model.hparams.data_dir, [score_func_name], "val")
    
    # Extract codes
    model.eval()
    train_codes = extract_codes(model, "train_labeled")
    val_codes = extract_codes(model, "val")
    
    # Train regression model on the extracted codes
    if regression_model_name in ["linear", "mlp"]:
        regression_model = train_nn(
            train_codes, val_codes, train_score_dataset, val_score_dataset, score_func_name, run
            )
    elif regression_model_name in ["gp"]:
        regression_model = train_gp(
            train_codes, val_codes, train_score_dataset, val_score_dataset, score_func_name, run
        )
    
    # Prepare code optimization
    train_scores = train_score_dataset.tsrs.squeeze(1)
    print(train_scores.size())
    topk_idxs = torch.topk(train_scores, k=k, largest=False)[1] # Start from lowest
    codes = torch.nn.Parameter(train_codes[topk_idxs].cuda())
    codes.requires_grad = True
    optimizer = torch.optim.SGD([codes], lr=1e-3)
    
    # Run gradopt
    for step in tqdm(range(steps)):
        loss = regression_model.neg_score(codes).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Project to e.g., hypersphere
        codes.data = model.ae.project(codes).data
        
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                smiles_list = model.ae.decoder.decode_smiles(codes.cuda(), deterministic=True)

            scores = torch.FloatTensor(score_func(smiles_list))
            clean_scores = scores[scores > corrupt_score + 1e-3]
            clean_ratio = clean_scores.size(0) / scores.size(0)

            statistics = dict()
            statistics["loss"] = loss
            statistics["clean_ratio"] = clean_ratio
            if clean_ratio > 1e-6:
                statistics["score/max"] = clean_scores.max()
                statistics["score/mean"] = clean_scores.mean()
            else:
                statistics["score/max"] = 0.0
                statistics["score/mean"] = 0.0

            for key, val in statistics.items():
                run[f"lso/gradopt/{regression_model_name}/{score_func_name}/{key}"].log(val)
