import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import neptune.new as neptune

from ae.module import AutoEncoderModule
from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset
from data.score.dataset import load_scores
from data.util import ZipDataset
from data.score.factory import get_scoring_func
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
        
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
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=hparams.num_workers,
    )

    codes = []
    for batched_data in tqdm(dataloader):
        if hparams.encoder_type == "seq":
            batched_data = tuple([item.cuda() for item in batched_data])
        elif hparams.encoder_type == "graph":
            batched_data = batched_data.cuda()
            
        with torch.no_grad():
            encoder_out = model.compute_encoder_out(batched_data)
            _, _, batched_codes = model.compute_codes(encoder_out)
            
        codes.append(batched_codes.detach().cpu())
    
    codes = torch.cat(codes, dim=0)
    return codes

def train_gp(gp, likelihood, train_codes, train_scores, val_codes, val_scores, run):
    likelihood = likelihood.cuda()
    gp = gp.cuda()
    
    likelihood.train()
    gp.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    train_codes = train_codes.cuda()
    train_scores = train_scores.cuda()
    for _ in tqdm(range(10000)):
        output = gp(train_codes)
        loss = -mll(output, train_scores).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        run["lso_gp/train/loss/gp_mll"].log(loss.item())

    gp.eval()
    likelihood.eval()
    
    val_codes = val_codes.cuda()
    val_scores = val_scores.cuda()

    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
            preds = gp(val_codes)
        
    run["lso_gp/validation/loss/gp_mse"] = torch.nn.functional.mse_loss(preds.mean, val_scores)
    
def neg_ei(codes, gp, fmin):
    dist = torch.distributions.Normal(loc=0., scale=1.)
    preds = gp(codes)
    mu, sigma = preds.mean, preds.stddev
    z = (fmin - mu) / sigma

    ei = ((fmin - mu) * dist.cdf(z) + sigma * dist.log_prob(z).exp())
    return -ei

def gp_opt_codes(model, gp, train_codes, train_scores, scoring_func_name, run):
    _, score_func, corrupt_score = get_scoring_func(scoring_func_name)

    fmin = torch.quantile(train_scores, 0.1)
    topk_idxs = torch.topk(train_scores, k=1024, largest=True)[1]

    codes = train_codes[topk_idxs].detach()
    codes = torch.nn.Parameter(codes)
    codes.requires_grad = True
    optimizer = torch.optim.SGD([codes], lr=1e-2)
    for step in tqdm(range(1000)):
        loss = neg_ei(codes.cuda(), gp, fmin).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model.ae_type in ["sae", "cae"]:
            codes.data.copy(F.normalize(codes, p=2, dim=1).data)


        if (step + 1) % 10 == 0:
            with torch.no_grad():
                smiles_list = model.decoder.decode_smiles(codes, deterministic=True)

            scores = torch.FloatTensor(score_func(smiles_list))
            clean_scores = scores[scores > corrupt_score + 1e-3]
            clean_ratio = clean_scores.size(0) / scores.size(0)

            statistics = dict()
            statistics["loss"] = loss
            statistics["clean_ratio"] = clean_ratio
            if clean_ratio > 0.0:
                statistics["score/max"] = clean_scores.max()
                statistics["score/mean"] = clean_scores.mean()
            else:
                statistics["score/max"] = 0.0
                statistics["score/mean"] = 0.0

            for key, val in statistics.items():
                run[f"lso_gp/{scoring_func_name}/{key}"].log(val)

def run_lso_gp(model, score_func_name, run):
    model.eval()
    train_codes = extract_codes(model, "train_labeled")
    train_scores = load_scores(model.hparams.data_dir, score_func_name, "train_labeled").view(-1)
    train_scores = (train_scores - train_scores.mean()) / train_scores.std()
    val_codes = extract_codes(model, "val")
    val_scores = load_scores(model.hparams.data_dir, score_func_name, "val").view(-1)
    val_scores = (val_scores - train_scores.mean()) / train_scores.std()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GPRegressionModel(train_codes, train_scores, likelihood)

    train_gp(gp, likelihood, train_codes, train_scores, val_codes, val_scores, run)

    gp_opt_codes(model, gp, train_codes, train_scores, score_func_name, run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", nargs="+", type=str, default=[])
    args = parser.parse_args()

    run = neptune.init(
        project="sungsahn0215/molrep",
        name="run_lso_gp",
        source_files=["*.py", "**/*.py"],
        tags=args.tag,
    )
    
    ae = AutoEncoderModule.load_from_checkpoint(args.ae_checkpoint_path)
    ae = ae.cuda()
    run_lso_gp(ae, "logp", run)