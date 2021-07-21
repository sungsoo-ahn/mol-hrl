from data.selfie.dataset import SelfieDataset
from data.smiles.dataset import SmilesDataset
import os
import pandas as pd

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data.graph.dataset import GraphDataset
from data.score.dataset import ScoreDataset
from data.score.factory import get_scoring_func

from lso.nn import train_nn
from lso.gp import train_gp

from neptune.new.types import File

def extract_codes(model, split):
    hparams = model.hparams
    if hparams.encoder_type == "graph":
        dataset = GraphDataset(hparams.data_dir, split=split)
    elif hparams.encoder_type == "smiles":
        dataset = SmilesDataset(hparams.data_dir, split=split)
    elif hparams.encoder_type == "selfie":
        dataset = SelfieDataset(hparams.data_dir, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=hparams.num_workers,
    )

    codes = []
    for batched_data in tqdm(dataloader):
        try:
            batched_data = batched_data.cuda()    
        except:
            batched_data = [i.cuda() for i in batched_data]
            
        with torch.no_grad():
            batched_codes = model.autoencoder.encoder(batched_data)
        codes.append(batched_codes.detach().cpu())
    
    codes = torch.cat(codes, dim=0)
    return codes

def run_gradopt(
    model, 
    regression_model_name, 
    score_func_name, 
    run, 
    k=1024, 
    steps=1000
    ):
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
    topk_idxs = torch.topk(train_scores, k=k, largest=True)[1] # Start from largest
    codes = train_codes[topk_idxs].cuda()
    codes.requires_grad = True

    # Run gradopt
    lr = 1e-3

    smiles_traj, scores_traj = [], []    

    for step in tqdm(range(steps)):
        loss = regression_model.neg_score(codes).sum()
        codes_grad = torch.autograd.grad(loss, codes, retain_graph=False, create_graph=False)[0]
        
        # Project to e.g., hypersphere
        codes.data = codes.data - lr * codes_grad.sign()

        if (step + 1) % 10 == 0:
            with torch.no_grad():
                smiles_list = model.autoencoder.decoder.sample_smiles(codes.cuda(), argmax=True)
            
            smiles_traj.append(smiles_list)
            scores = torch.FloatTensor(score_func(smiles_list))
            scores_traj.append(scores)
            
            scores_traj_max = torch.stack(scores_traj, dim=0).max(dim=0)[0]
            
            statistics = dict()
            statistics["score"] = scores_traj_max.max()
            #statistics["score/std"] = scores_traj_max.std()    

            clean_scores = scores[scores > corrupt_score + 1e-3]
            clean_ratio = clean_scores.size(0) / scores.size(0)
            statistics["clean_ratio"] = clean_ratio

            for key, val in statistics.items():
                run[f"lso/gradopt/{regression_model_name}/{score_func_name}/{key}"].log(val)


    df = pd.DataFrame(data=smiles_traj)
    log_dir = run["log_dir"].fetch()
    filename = os.path.join(
        log_dir, f"gradopt_{regression_model_name}_{score_func_name}_smiles_traj.csv"
        )
    df.to_csv(filename)
    run[f"gradopt_{regression_model_name}_{score_func_name}_smiles_traj"].upload(File(filename))