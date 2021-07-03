from collections import defaultdict
import argparse
import rdkit
from rdkit.Chem import Draw

import torch
import pytorch_lightning as pl
import neptune.new as neptune
from neptune.new.types import File


from model.ae import AutoEncoderModel
from model.vae import VariationalAutoEncoderModel
from data.score.factory import get_scoring_func

from tqdm import tqdm

def compute_init_codes(model):
    return torch.randn(1024, hparams.code_dim)
    
def optimize_codes_linear(model, run, hparams):
    init_codes = compute_init_codes(model, num_codes=hparams.num_codes)
    codes = torch.nn.Parameter(init_codes).to(model.device)
    optim = torch.optim.SGD([codes], lr=hparams.code_lr)

    smiles_traj_list = defaultdict(list)
    _, parallel_score_func, corrupt_score = get_scoring_func("penalized_logp", num_workers=32)
    for step in tqdm(range(hparams.num_code_opt_steps)):
        pred_scores = model.predict_score(codes)
        loss = -pred_scores.sum()
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        smiles_list = model.decode_deterministic(codes)
        scores = torch.FloatTensor(parallel_score_func(smiles_list))
        clean_scores = scores[scores > corrupt_score + 1e-3]
        clean_ratio = clean_scores.size(0) / scores.size(0)
        run["clean_ratio"].log(clean_ratio)
        if clean_scores.size(0) > 0:
            run["score/max"].log(clean_scores.max())
            run["score/mean"].log(clean_scores.mean())
            run["pred/max"].log(pred_scores.max())
            run["pred/mean"].log(pred_scores.mean())

        for idx, (smiles, score) in enumerate(zip(smiles_list, scores)):
            if smiles_traj_list[idx][-1] != smiles and score > corrupt_score + 1e-3:
                smiles_traj_list[idx].append(smiles)

    smiles_traj_list = sorted(smiles_traj_list, key=len)

    for smile in smiles_traj_list[-1]:
        try:
            draw_mol = rdkit.Chem.MolFromSmiles(smile)
            run[f"mol"].log(File.as_image(Draw.MolToImage(draw_mol)))
        except:
            pass
        
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--model_type", type=str)
    args = parser.parse_args()
    if args.model_type == "ae":
        model = AutoEncoderModel.load_from_checkpoint(args.load_path)
    else:
        model = VariationalAutoEncoderModel.load_from_checkpoint(args.load_path)
    hparams = model.hparams

    run = neptune.init(project="sungsahn0215/mol-hrl", name="neptune_logs")
    codes = torch.nn.Parameter(torch.randn(1024, hparams.code_dim).cuda())
    model = model.cuda()
    optim = torch.optim.SGD([codes], lr=1e-2)

    smiles_traj_list = [[""] for _ in range(1024)]
    for step in tqdm(range(100)):
        if args.model_type == "vae" and hparams.spherical:
            codes = torch.nn.functional.normalize(codes, p=2, dim=1)

        pred_scores = model.scores_predictor(codes)
        loss = -pred_scores[:, 0].sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        smiles_list = model.decode_deterministic(codes)
        _, parallel_score_func, corrupt_score = get_scoring_func("penalized_logp", num_workers=32)
        scores = torch.FloatTensor(parallel_score_func(smiles_list))
        clean_scores = scores[scores > corrupt_score + 1e-3]
        clean_ratio = clean_scores.size(0) / scores.size(0)
        run["clean_ratio"].log(clean_ratio)
        if clean_scores.size(0) > 0:
            run["score/max"].log(clean_scores.max())
            run["score/mean"].log(clean_scores.mean())
            run["pred/max"].log(pred_scores.max())
            run["pred/mean"].log(pred_scores.mean())

        for idx, (smiles, score) in enumerate(zip(smiles_list, scores)):
            if smiles_traj_list[idx][-1] != smiles and score > corrupt_score + 1e-3:
                smiles_traj_list[idx].append(smiles)

    smiles_traj_list = sorted(smiles_traj_list, key=len)

    for smile in smiles_traj_list[-1]:
        try:
            draw_mol = rdkit.Chem.MolFromSmiles(smile)
            run[f"mol"].log(File.as_image(Draw.MolToImage(draw_mol)))
        except:
            pass