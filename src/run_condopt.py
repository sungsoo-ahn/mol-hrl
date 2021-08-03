from tqdm import tqdm
import argparse
from data.util import ZipDataset
from data.sequence.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
import torch
import numpy as np
import neptune.new as neptune

from module.decoder.sequence import SequenceDecoder
from data.score.factory import get_scoring_func

class QueueDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, score_list, k):
        self.smiles_list = smiles_list
        self.score_list = score_list
        self.k = k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--code_dim", type=int, default=256)
    parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
    parser.add_argument("--decoder_num_layers", type=int, default=3)
    parser.add_argument("--decoder_max_length", type=int, default=512)
    parser.add_argument("--train_split", type=str, default="train_01")
    parser.add_argument("--scoring_func_name", type=str, default="penalized_logp")
    parser.add_argument("--num_stages", type=int, default=100)
    parser.add_argument("--num_queries_per_stage", type=int, default=1)
    parser.add_argument("--reweight_k", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=float, default=256)
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--num_steps_per_stage", type=int, default=50)
    parser.add_argument("--tag", type=str, default="notag")
    hparams = parser.parse_args()

    device = torch.device(0)
    decoder = SequenceDecoder(hparams)
    cond_embedding = torch.nn.Linear(1, hparams.code_dim)
    if hparams.load_checkpoint_path != "":
        state_dict = torch.load(hparams.load_checkpoint_path)
        decoder.load_state_dict(state_dict["decoder"])
        cond_embedding.load_state_dict(state_dict["cond_embedding"])

    decoder.to(device)
    cond_embedding.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(cond_embedding.parameters()), lr=1e-3)
    
    _, scoring_func, corrupt_score = get_scoring_func(hparams.scoring_func_name)
    sequence_dataset = SequenceDataset(hparams.data_dir, hparams.train_split)
    score_dataset = ScoreDataset(hparams.data_dir, hparams.scoring_func_name, hparams.train_split)

    def sample(queries):
        codes = cond_embedding(queries)
        with torch.no_grad():
            smiles_list = decoder.sample_smiles(codes, argmax=False)

        score_list = scoring_func(smiles_list)
        valid_idxs = [idx for idx, score in enumerate(score_list) if score > corrupt_score]
        valid_score_list = [score_list[idx] for idx in valid_idxs]
        valid_smiles_list = [smiles_list[idx] for idx in valid_idxs]
        return valid_smiles_list, valid_score_list

    def get_queries(num_queries):
        query = score_dataset.raw_tsrs.max() + 2.0
        queries = query.view(1, 1).expand(num_queries, 1)
        queries = score_dataset.normalize(queries)
        return queries
        
    def run_steps(num_steps):
        dataset = ZipDataset(score_dataset, sequence_dataset)
        scores_np = score_dataset.raw_tsrs.view(-1).numpy()
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (hparams.reweight_k * len(scores_np) + ranks)
        #print(weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, 
            batch_size=hparams.train_batch_size, 
            collate_fn=dataset.collate
            )
        
        step = 0
        while step < num_steps:
            step += 1
            try: 
                batched_data = next(data_iter)
            except:
                data_iter = iter(loader)
                batched_data = next(data_iter)
            
            batched_cond_data, batched_target_data = batched_data
            batched_cond_data = batched_cond_data.to(device)
            batched_target_data = [tsr.to(device) for tsr in batched_target_data]
            codes = cond_embedding(batched_cond_data)
            decoder_out = decoder(batched_target_data, codes)
            loss, _ = decoder.compute_recon_loss(decoder_out, batched_target_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run["train/loss"].log(loss)

    run = neptune.init(
        project="sungsahn0215/molrep", name="run_condopt", source_files=["*.py", "**/*.py"], tags=[hparams.tag]
        )
    
    for stage in tqdm(range(hparams.num_stages)):
        #
        decoder.train()
        cond_embedding.train()
        run_steps(hparams.num_steps_per_stage if stage > 0 else hparams.num_warmup_steps)
        #
        decoder.eval()
        cond_embedding.eval()
        queries = get_queries(max(hparams.num_queries_per_stage, 128))
        queries = queries.to(device)
        
        smiles_list, score_list = [], []
        for _ in range(10):
            new_smiles_list, new_score_list = sample(queries)
            smiles_list.extend(new_smiles_list)
            score_list.extend(new_score_list)
            if len(smiles_list) > hparams.num_queries_per_stage:
                smiles_list = smiles_list[:hparams.num_queries_per_stage]
                score_list = score_list[:hparams.num_queries_per_stage]
                break
        #
        sequence_dataset.update(smiles_list)
        score_dataset.update(score_list)

        top123 = torch.topk(score_dataset.raw_tsrs.view(-1), k=3)[0]
        run["top1"].log(top123[0])
        run["top2"].log(top123[1])
        run["top3"].log(top123[2]) 
        