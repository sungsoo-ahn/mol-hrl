from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset
from data.score.factory import get_scoring_func

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
            
        with torch.no_grad():
            batched_codes = model.ae.encode(batched_data)
            
        codes.append(batched_codes.detach().cpu())
    
    codes = torch.cat(codes, dim=0)
    return codes

def run_lso(model, regression_model, train_codes, train_scores, scoring_func_name, run):
    _, score_func, corrupt_score = get_scoring_func(scoring_func_name)

    topk_idxs = torch.topk(train_scores, k=1024, largest=True)[1]

    codes = train_codes[topk_idxs].detach()
    codes = torch.nn.Parameter(codes.cuda())
    codes.requires_grad = True
    optimizer = torch.optim.SGD([codes], lr=1e-2)
    for step in tqdm(range(1000)):
        loss = regression_model.neg_score(codes).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model.hparams.ae_type in ["sae", "cae"]:
            codes.data = F.normalize(codes, p=2, dim=1).data

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
                run[f"lso/{regression_model.tag}/{scoring_func_name}/{key}"].log(val)