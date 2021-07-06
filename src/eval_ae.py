import argparse
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from data.graph.dataset import GraphDataset
from data.seq.dataset import SequenceDataset
from ae.module import AutoEncoderModule
from data.smiles.util import load_smiles_list

import neptune.new as neptune


def compute_code_dists(model):
    if model.hparams.encoder_type == "seq":
        input_dataset = SequenceDataset(model.hparams.data_dir, "train_labeled")
    elif model.hparams.encoder_type == "graph":
        input_dataset = GraphDataset(model.hparams.data_dir, "train_labeled")

    loader = DataLoader(
        input_dataset,
        batch_size=model.hparams.batch_size,
        shuffle=False,
        num_workers=model.hparams.num_workers,
        collate_fn=input_dataset.collate_fn,
    )

    codes_list = []
    for batched_input_data in tqdm(loader):
        try:
            batched_input_data = batched_input_data.cuda()
        except:
            batched_input_data = (
                batched_input_data[0].cuda(),
                batched_input_data[1].cuda(),
            )

        with torch.no_grad():
            encoder_out = model.compute_encoder_out(batched_input_data)
            codes = model.compute_codes(encoder_out)[-1]

        codes_list.append(codes)

    codes = torch.cat(codes_list, dim=0)
    code_dists = torch.cdist(codes, codes, p=2)
    return code_dists


def eval_knn(model, k_list=[50, 10, 5, 1]):
    code_dists = compute_code_dists(model)
    smiles_list = load_smiles_list(model.hparams.data_dir, split="train_labeled")
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = [
        FingerprintMols.FingerprintMol(
            mol,
            minPath=1,
            maxPath=7,
            fpSize=2048,
            bitsPerHash=2,
            useHs=True,
            tgtDensity=0.0,
            minSize=128,
        )
        for mol in mols
    ]
    statistics = dict()
    for k in k_list:
        neighbors_list = torch.topk(code_dists, k=k + 1, dim=1, largest=False)[
            1
        ].tolist()
        sim_avg = 0.0
        for idx, neighbors in enumerate(neighbors_list):
            assert idx == neighbors[0]
            query_fp = fps[idx]
            target_fps = [fps[idx1] for idx1 in neighbors[1:]]
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, target_fps)
            sim_avg += torch.tensor(sims).mean() / len(mols)

        statistics[f"knn_tanimoto/{k:02d}"] = sim_avg

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", nargs="+", type=str, default=[])
    args = parser.parse_args()

    model = AutoEncoderModule.load_from_checkpoint(args.ae_checkpoint_path)
    model = model.cuda()
    model.eval()
    statistics = eval_knn(model)

    run = neptune.init(
        project="sungsahn0215/molrep",
        name="eval_ae",
        source_files=["*.py", "**/*.py"],
        tags=args.tag,
    )
    run["parameters"] = vars(args)
    run["statistics"] = statistics
