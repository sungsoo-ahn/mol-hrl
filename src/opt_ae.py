import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from model.ae import AutoEncoderModel
from model.vae import VariationalAutoEncoderModel

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
    
    codes = torch.nn.Parameter(torch.randn(1024, hparams.code_dim).cuda())
    model = model.cuda()
    optim = torch.optim.SGD([codes], lr=1e-3)

    for step in range(1):
        pred_scores = model.scores_predictor(codes)
        loss = pred_scores[:, 0].sum()
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        sequences, lengths, _ = model.decoder.argmax_sample(
            codes, 
            model.vocabulary.get_start_id(), 
            model.vocabulary.get_end_id(), 
            model.vocabulary.get_max_length()
            )