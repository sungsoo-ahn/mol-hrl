import argparse
import rdkit
from rdkit.Chem import Draw
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

import neptune.new as neptune
from neptune.new.types import File


from model.ae import AutoEncoderModel
from model.vae import VariationalAutoEncoderModel
from data.sequence.util import string_from_sequence
from data.score.factory import get_scoring_func

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #AutoEncoderModel.add_args(parser)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--model_type", type=str)
    args = parser.parse_args()
    if args.model_type == "ae":
        model = AutoEncoderModel.load_from_checkpoint(args.load_path)
    else:
        model = VariationalAutoEncoderModel.load_from_checkpoint(args.load_path)

    hparams = model.hparams

    #neptune_logger = NeptuneLogger(
    #    project_name="sungsahn0215/mol-hrl", experiment_name="neptune_logs", params=vars(hparams),
    #)

    #checkpoint_callback = ModelCheckpoint(monitor='train/loss/total')
    #trainer = pl.Trainer(
    #    gpus=1, 
    #    logger=neptune_logger,
    #    default_root_dir="../resource/log/", 
    #    max_epochs=100,
    #    callbacks=[checkpoint_callback]
    #)
    #trainer.fit(model)

    run = neptune.init(project="sungsahn0215/mol-hrl", name="neptune_logs")

    codes = torch.nn.Parameter(torch.randn(1024, hparams.code_dim).cuda())
    model = model.cuda()
    optim = torch.optim.SGD([codes], lr=1e-2)

    smiles_traj_list = [[""] for _ in range(1024)]
    for step in tqdm(range(10)):
        if args.model_type == "vae" and hparams.spherical:
            codes = torch.nn.functional.normalize(codes, p=2, dim=1)
        pred_scores = model.scores_predictor(codes)
        loss = -pred_scores[:, 0].sum()
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        sequences, lengths, _ = model.decoder.argmax_sample(
            codes, 
            model.vocabulary.get_start_id(), 
            model.vocabulary.get_end_id(), 
            model.vocabulary.get_max_length()
            )
        
        sequences = sequences.cpu().split(1, dim=0)
        lengths = lengths.cpu()
        sequences = [sequence[:length] for sequence, length in zip(sequences, lengths)]

        smiles_list = [
            string_from_sequence(sequence, model.tokenizer, model.vocabulary)
            for sequence in sequences
        ]
        
        _, parallel_score_func, corrupt_score = get_scoring_func("penalized_logp", num_workers=32)
        scores = torch.FloatTensor(parallel_score_func(smiles_list))
        clean_scores = scores[scores > corrupt_score + 1e-3]
        
        clean_ratio = clean_scores.size(0) / scores.size(0)
        run["clean_ratio"].log(clean_ratio)
        if clean_scores.size(0) > 0:
            run["score/max"].log(clean_scores.max())
            run["score/mean"].log(clean_scores.mean())
            run["pred/max"].log(pred_scores.max())
            run["pred/mean"].log(pred_scores.max())
        
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
        