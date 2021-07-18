from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from data.score.factory import get_scoring_func
from data.score.dataset import ScoreDataset
from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset

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
    for batched_data in dataloader:
        if hparams.encoder_type == "seq":
            batched_data = tuple([item.cuda() for item in batched_data])
        elif hparams.encoder_type == "graph":
            batched_data = batched_data.cuda()

        with torch.no_grad():
            batched_codes = model.ae.encode(batched_data)
            
        codes.append(batched_codes.detach().cpu())
    
    codes = torch.cat(codes, dim=0)
    return codes


def run_bo(model, score_func_name, run):
    #
    BATCH_SIZE = 1
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256

    seed=1
    torch.manual_seed(seed)
    ae = model.ae.cuda()
    device = model.device
    
    # setup scoring function
    _, smiles_score_func, corrupt_score = get_scoring_func(score_func_name)
    invalid_scores = ScoreDataset(model.hparams.data_dir, [score_func_name], "train").tsrs.min()
    def score(codes):
        smiles_list = ae.decoder.decode_smiles(codes.to(device), deterministic=True)
        scores = torch.FloatTensor(smiles_score_func(smiles_list))
        scores[scores < corrupt_score + 1] = invalid_scores
        return scores.unsqueeze(1)
        
    # initialize 
    train_codes = extract_codes(model, "train_labeled")[:1024].to(device)
    train_scores = score(train_codes).to(device)
    best_score = train_scores.max().item()
    code_dim = train_codes.size(0)
    bounds = [train_codes.min(dim=1)[0], train_codes.max(dim=1)[0]]

    #
    def get_fitted_model(train_codes, train_scores, state_dict=None):
        # initialize and fit model
        model = SingleTaskGP(
            train_X=normalize(train_codes, bounds=bounds), train_Y=train_scores
            )
        if state_dict is not None:
            model.load_state_dict(state_dict)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(device)
        fit_gpytorch_model(mll)
        return model
    
    #
    def optimize_acqf_and_get_observation(acq_func):
        # optimize
        new_codes, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(code_dim, device=device), 
                torch.ones(code_dim, device=device),
            ]),
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        # observe new values
        new_codes = unnormalize(new_codes, bounds=bounds).to(device)
        new_scores = score(new_codes).to(device)
        return new_codes, new_scores
    
    N_BATCH = 1024
    MC_SAMPLES = 2048
    
    # call helper function to initialize model
    best_observed = [best_score]
    state_dict = None
    
    run["best_observed"].log(best_score)
    for iteration in tqdm(range(N_BATCH)):    
        # fit the model
        model = get_fitted_model(
            train_codes, 
            standardize(train_scores), 
            state_dict=state_dict,
        )
        
        # define the qNEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
        qEI = qExpectedImprovement(
            model=model, 
            sampler=qmc_sampler, 
            best_f=torch.quantile(standardize(train_scores), q=0.9)
            )

        # optimize and get new observation
        new_codes, new_scores = optimize_acqf_and_get_observation(qEI)

        # update training points
        train_codes = torch.cat((train_codes, new_codes))
        train_scores = torch.cat((train_scores, new_scores))

        # update progress
        best_value = train_scores.max().item()
        best_observed.append(best_value)
        
        state_dict = model.state_dict()

        run["best_observed"].log(best_value)