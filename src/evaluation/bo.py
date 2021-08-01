from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from gpytorch.kernels import LinearKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from module.pl_autoencoder import AutoEncoderModule
from data.score.factory import get_scoring_func
from data.score.dataset import ScoreDataset
from data.graph.dataset import GraphDataset
from data.sequence.dataset import SequenceDataset

from data.smiles.util import is_valid_smiles


def extract_codes(model, split):
    hparams = model.hparams
    if hparams.encoder_type == "graph":
        dataset = GraphDataset(hparams.data_dir, split=split)
    elif hparams.encoder_type == "smiles":
        dataset = SequenceDataset(hparams.data_dir, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=hparams.num_workers,
    )

    codes = []
    for batched_data in dataloader:
        try:
            batched_data = batched_data.cuda()
        except:
            batched_data = [i.cuda() for i in batched_data]

        with torch.no_grad():
            batched_codes = model.autoencoder.encode(batched_data)

        codes.append(batched_codes.detach().cpu())

    codes = torch.cat(codes, dim=0)
    return codes


def run_bo(checkpoint_path, score_func_name, run, covar_module):
    BATCH_SIZE = 10
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256
    N_BATCH = 200
    MC_SAMPLES = 2048
    NUM_REPS = 5

    #
    model = AutoEncoderModule.load_from_checkpoint(checkpoint_path)
    model = model.cuda()
    model.eval()
    ae = model.autoencoder
    device = model.device

    #
    dataset_codes = extract_codes(model, "train")
    dataset_scores = ScoreDataset(model.hparams.data_dir, [score_func_name], "train").raw_tsrs

    # setup scoring function
    _, smiles_score_func, corrupt_score = get_scoring_func(score_func_name)
    invalid_scores = dataset_scores.min()

    code_dim = dataset_codes.size(1)

    def score(codes):
        smiles_list = ae.decoder.sample_smiles(codes.to(device), argmax=True)
        scores = smiles_score_func(smiles_list)
        scores = torch.FloatTensor(scores)
        scores[scores < corrupt_score + 1e-6] = invalid_scores
        return scores.unsqueeze(1)

    def get_fitted_model(train_codes, train_scores, bounds, state_dict=None):
        # initialize and fit model
        if covar_module == "rbf":
            model = SingleTaskGP(train_X=normalize(train_codes, bounds=bounds), train_Y=train_scores,)

        elif covar_module == "linear":
            model = SingleTaskGP(
                train_X=normalize(train_codes, bounds=bounds), train_Y=train_scores, covar_module=LinearKernel()
            )

        if state_dict is not None:
            model.load_state_dict(state_dict)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(device)
        if covar_module == "rbf":
            fit_gpytorch_model(mll, max_retries=10)
        elif covar_module == "linear":
            fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, max_retries=10)

        return model

    #
    def optimize_acqf_and_get_observation(acq_func, bounds):
        # optimize
        new_codes, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(code_dim, device=device), torch.ones(code_dim, device=device),]),
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        # observe new values
        new_codes = unnormalize(new_codes, bounds=bounds).to(device)
        new_scores = score(new_codes).to(device)
        return new_codes, new_scores

    def run_single_bo(rep_id):
        # initialize
        train_idxs = torch.topk(dataset_scores.squeeze(1), k=1000)[1]
        train_scores = dataset_scores[train_idxs].to(device)
        train_codes = dataset_codes[train_idxs].to(device)

        best_score = train_scores.max().item()

        # call helper function to initialize model
        best_observed = [best_score]
        state_dict = None

        run[f"{score_func_name}/rep{rep_id}/best_observed"].log(best_score)
        for _ in tqdm(range(N_BATCH)):
            # fit the model
            bounds = [torch.min(train_codes, dim=0)[0], torch.max(train_codes, dim=0)[0]]
            model = get_fitted_model(train_codes, standardize(train_scores), bounds, state_dict=state_dict)

            # define the qNEI acquisition module using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_scores).max())

            # optimize and get new observation
            new_codes, new_scores = optimize_acqf_and_get_observation(qEI, bounds)

            # update training points
            train_codes = torch.cat([train_codes, new_codes], dim=0)
            train_scores = torch.cat([train_scores, new_scores], dim=0)

            # update progress
            best_value = train_scores.max().item()
            best_observed.append(best_value)

            state_dict = model.state_dict()

            run[f"{score_func_name}/rep{rep_id}/best_observed"].log(best_value)
            run[f"{score_func_name}/rep{rep_id}/observed"].log(new_scores.max().item())

        train_scores = train_scores.squeeze(1)
        top1 = torch.topk(train_scores, k=1)[0].mean().item()
        top10 = torch.topk(train_scores, k=10)[0].mean().item()
        top100 = torch.topk(train_scores, k=100)[0].mean().item()

        return top1, top10, top100

    top1s, top10s, top100s = [], [], []
    fail_cnt = 0
    for rep_id in range(NUM_REPS):
        try:
            top1, top10, top100 = run_single_bo(rep_id)
            top1s.append(top1)
            top10s.append(top10)
            top100s.append(top100)

            run[f"{score_func_name}/top1/avg"] = np.mean(top1s)
            run[f"{score_func_name}/top10/avg"] = np.mean(top10s)
            run[f"{score_func_name}/top100/avg"] = np.mean(top100s)
            run[f"{score_func_name}/top1/std"] = np.std(top1s)
            run[f"{score_func_name}/top10/std"] = np.std(top10s)
            run[f"{score_func_name}/top100/std"] = np.std(top100s)
        except:
            fail_cnt += 1
            run[f"{score_func_name}/fail_cnt"] = fail_cnt
