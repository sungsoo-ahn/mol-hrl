import torch
import numpy as np
import neptune.new as neptune
from data.smiles.util import load_smiles_list
from tqdm import tqdm
import argparse

from data.sequence.dataset import SequenceDataset
from data.score.dataset import ScoreDataset
from data.graph.dataset import GraphDataset
from data.util import ZipDataset

from module.decoder.sequence import SequenceDecoder
from module.encoder.graph import GraphEncoder
from module.plug import PlugVariationalAutoEncoder
from data.score.factory import get_scoring_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
    parser.add_argument("--load_checkpoint_path", type=str, default="../resource/checkpoint/default_codedecoder.pth")
    parser.add_argument("--code_dim", type=int, default=256)
    
    # GraphEncoder specific
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_num_layers", type=int, default=5)

    # SequentialDecoder specific
    parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
    parser.add_argument("--decoder_num_layers", type=int, default=3)
    parser.add_argument("--decoder_max_length", type=int, default=512)

    #
    parser.add_argument("--plug_code_dim", type=int, default=64)
    parser.add_argument("--plug_beta", type=float, default=0.01)
    parser.add_argument("--plug_width_factor", type=float, default=1.0)
        
    #
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--cond_embedding_mlp", action="store_true")
    parser.add_argument("--train_split", type=str, default="train_001")
    parser.add_argument("--scoring_func_name", type=str, default="penalized_logp")
    parser.add_argument("--num_stages", type=int, default=1000)
    parser.add_argument("--num_queries_per_stage", type=int, default=1)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--reweight_k", type=float, default=1e-2)
    parser.add_argument("--train_batch_size", type=float, default=256)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--num_steps_per_stage", type=int, default=10)
    parser.add_argument("--tags", type=str, nargs="+", default=[])
    hparams = parser.parse_args()

    device = torch.device(0)
    encoder = GraphEncoder(hparams)
    decoder = SequenceDecoder(hparams)
    plug_vae = PlugVariationalAutoEncoder(hparams)
    
    if hparams.load_checkpoint_path != "":
        state_dict = torch.load(hparams.load_checkpoint_path)
        encoder.load_state_dict(state_dict["encoder"])
        decoder.load_state_dict(state_dict["decoder"])
        plug_vae.load_state_dict(state_dict["plug_vae"])
        
    encoder.to(device)
    decoder.to(device)
    plug_vae.to(device)

    params = list(plug_vae.parameters())
    params += list(encoder.parameters())
    params += list(decoder.parameters())
    
    #if not hparams.tune_all:
        
    optimizer = torch.optim.Adam(params, lr=1e-5)
    
    _, scoring_func, corrupt_score = get_scoring_func(hparams.scoring_func_name)
    graph_dataset = GraphDataset(hparams.data_dir, hparams.train_split)
    sequence_dataset = SequenceDataset(hparams.data_dir, hparams.train_split)
    score_dataset = ScoreDataset(hparams.data_dir, hparams.scoring_func_name, hparams.train_split)

    def sample(queries):
        codes = plug_vae.sample(queries)
        with torch.no_grad():
            smiles_list = decoder.sample_smiles(codes, argmax=False)

        score_list = scoring_func(smiles_list)
        valid_idxs = [idx for idx, score in enumerate(score_list) if score > corrupt_score]
        valid_score_list = [score_list[idx] for idx in valid_idxs]
        valid_smiles_list = [smiles_list[idx] for idx in valid_idxs]
        
        return valid_smiles_list, valid_score_list

    def get_queries(num_queries):
        query = score_dataset.raw_tsrs.max() + 2.0
        run["query"].log(query)
        queries = query.view(1, 1).expand(num_queries, 1)
        queries = score_dataset.normalize(queries)
        return queries
        
    def run_steps(num_steps):
        dataset = ZipDataset(score_dataset, graph_dataset, sequence_dataset)
        
        scores_np = score_dataset.raw_tsrs.view(-1).numpy()
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (hparams.reweight_k * len(scores_np) + ranks)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(scores_np), replacement=True
            )
        loader = torch.utils.data.DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=hparams.train_batch_size, 
            collate_fn=dataset.collate,
            drop_last=True
            )
        
        #else:
        #    loader = torch.utils.data.DataLoader(
        #        dataset, 
        #        batch_size=hparams.train_batch_size, 
        #        collate_fn=dataset.collate,
        #        shuffle=True,
        #        drop_last=True
        #        )
        
        step = 0
        while step < num_steps:
            step += 1
            try: 
                batched_data = next(data_iter)
            except:
                data_iter = iter(loader)
                batched_data = next(data_iter)
            
            batched_cond_data, batched_input_data, batched_target_data = batched_data
            batched_cond_data = batched_cond_data.to(device)
            batched_input_data = batched_input_data.to(device)
            batched_target_data = [tsr.to(device) for tsr in batched_target_data]

            codes = encoder(batched_input_data)
            decoder_out = decoder(batched_target_data, codes)
            loss0, _ = decoder.compute_recon_loss(decoder_out, batched_target_data)
            loss1, _ = plug_vae.step(codes.detach(), batched_cond_data)

            loss = loss0 + loss1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()

            run["logs/train/loss/total"].log(loss)

    run = neptune.init(
        project="sungsahn0215/molrep", 
        name="run_condopt", 
        source_files=["*.py", "**/*.py"], 
        tags=["condopt"]+hparams.tags,
        )
    
    seen_scores_list = []
    seen_smiles_list = load_smiles_list(hparams.data_dir, hparams.train_split)
    for stage in tqdm(range(hparams.num_stages)):
        #
        decoder.train()
        encoder.train()
        plug_vae.train()
        run_steps(hparams.num_steps_per_stage if stage > 0 else hparams.num_warmup_steps)
        
        #
        decoder.eval()
        encoder.eval()
        plug_vae.eval()
        queries = get_queries(max(hparams.num_queries_per_stage, 128))
        queries = queries.to(device)
        
        smiles_list, score_list = [], []
        num_valids = 0
        for query_step in range(10):
            valid_smiles_list, valid_score_list = sample(queries)
            num_valids += len(valid_smiles_list)

            new_idxs = [idx for idx, smiles in enumerate(valid_smiles_list) if smiles not in seen_smiles_list]
            new_smiles_list = [valid_smiles_list[idx] for idx in new_idxs]
            new_score_list = [valid_score_list[idx] for idx in new_idxs]
            
            smiles_list.extend(new_smiles_list)
            score_list.extend(new_score_list)
            
            if len(smiles_list) + 1 > hparams.num_queries_per_stage:
                valid_ratio = float(num_valids) / (query_step + 1) / max(hparams.num_queries_per_stage, 128)
                run["valid_ratio"].log(valid_ratio)

                new_ratio = float(len(smiles_list)) / (query_step + 1) / max(hparams.num_queries_per_stage, 128)
                run["new_ratio"].log(new_ratio)

                smiles_list = smiles_list[:hparams.num_queries_per_stage]
                score_list = score_list[:hparams.num_queries_per_stage]
                break
        
        sequence_dataset.update(smiles_list)
        graph_dataset.update(smiles_list)
        score_dataset.update(score_list)
        seen_smiles_list = list(set(seen_smiles_list + new_smiles_list))
        seen_scores_list = seen_scores_list + score_list

        if len(seen_scores_list) > 2:
            top123 = torch.topk(torch.FloatTensor(seen_scores_list), k=3)[0]
            run["top1"].log(top123[0])
            run["top2"].log(top123[1])
            run["top3"].log(top123[2])
        