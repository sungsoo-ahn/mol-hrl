from tqdm import tqdm

import torch
import torch.nn.functional as F

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

from data.score.dataset import load_scores

class GPRegressionModel(gpytorch.models.ExactGP):
    tag="gp"
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def neg_score(self, x):
        dist = torch.distributions.Normal(loc=0., scale=1.)
        preds = self(x)
        mu, sigma = preds.mean, preds.stddev
        #return -mu
        z = (mu - self.fmax) / sigma
        ei = ((mu - self.fmax) * dist.cdf(z) + sigma * dist.log_prob(z).exp())
        return -ei


def train_gp(train_codes, val_codes, train_score_dataset, val_score_dataset, score_func_name, run, rep):
    train_scores = train_score_dataset.tsrs.squeeze(1)
    val_scores = val_score_dataset.tsrs.squeeze(1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GPRegressionModel(train_codes, train_scores, likelihood)
    
    likelihood = gp.likelihood.cuda()
    gp = gp.cuda()
    
    likelihood.train()
    gp.train()

    gp.fmax = torch.quantile(train_scores, 0.95)

    optimizer = torch.optim.Adam(gp.parameters(), lr=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    train_codes = train_codes.cuda()
    train_scores = train_scores.cuda()
    for step in tqdm(range(100)):
        output = gp(train_codes)
        loss = -mll(output, train_scores).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        run[f"rep{rep}/lso_gp/{score_func_name}/train/loss/gp_mll"].log(loss.item())

        if (step + 1) % 1 == 0:
            gp.eval()
            likelihood.eval()
            
            val_codes = val_codes.cuda()
            val_scores = val_scores.cuda()

            with torch.no_grad():
                preds = gp(val_codes)
                
            run[f"rep{rep}/lso_gp/{score_func_name}/validation/loss/gp_mse"].log(
                torch.nn.functional.mse_loss(preds.mean, val_scores)
            )

            gp.train()
            likelihood.train()
    
    gp.train()
    gp.eval()
    likelihood.train()
    likelihood.eval()
    
    return gp
    
"""
def run_lso_gp(model, score_func_name, run):
    model.eval()
    train_codes = extract_codes(model, "train_labeled")
    train_scores = load_scores(model.hparams.data_dir, score_func_name, "train_labeled").view(-1)
    train_scores = (train_scores - train_scores.mean()) / train_scores.std()
    val_codes = extract_codes(model, "val")
    val_scores = load_scores(model.hparams.data_dir, score_func_name, "val").view(-1)
    val_scores = (val_scores - train_scores.mean()) / train_scores.std()
    
    
    train_gp(gp, train_codes, train_scores, val_codes, val_scores, run)
    
    gp.eval()
    run_lso(model, gp, train_codes, train_scores, score_func_name, run)
"""