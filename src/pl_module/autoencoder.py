from argparse import Namespace
from re import S
from data.util import load_tokenizer

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from module.factory import load_encoder, load_decoder
from module.vq_layer import VectorQuantizeLayer
from data.factory import load_dataset, load_collate
from data.smiles.util import canonicalize


def compute_sequence_accuracy(logits, batched_sequence_data, pad_id=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == pad_id] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, pad_id=0):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id,
    )

    return loss


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.tokenizer = load_tokenizer()
        code_dim = hparams.code_dim * hparams.vq_code_dim if hparams.vq else hparams.code_dim
        self.encoder = load_encoder(hparams.encoder_name, code_dim)
        self.decoder = load_decoder(hparams.decoder_name, code_dim)
        if self.hparams.vq:
            self.vq_layer = VectorQuantizeLayer(hparams.vq_code_dim, hparams.vq_num_vocabs)

        self.train_dataset = load_dataset(hparams.dataset_name, hparams.task, "train")
        self.val_dataset = load_dataset(hparams.dataset_name, hparams.task, "valid")
        self.collate = load_collate(hparams.dataset_name)


    @staticmethod
    def add_args(parser):
        # optimizer
        parser.add_argument("--lr", type=float, default=1e-4)

        # dataloader
        parser.add_argument("--task", type=str, default="zinc")
        parser.add_argument("--dataset_name", type=str, default="graph2seq")
        
        # dataset
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        # models
        parser.add_argument("--encoder_name", type=str, default="gnn_base")
        parser.add_argument("--decoder_name", type=str, default="lstm_base")
        parser.add_argument("--vq", action="store_true")
        parser.add_argument("--vq_code_dim", type=int, default=128)
        parser.add_argument("--vq_num_vocabs", type=int, default=256)
        parser.add_argument("--code_dim", type=int, default=256)
        
        #
        parser.add_argument("--check_sample_quality_freq", type=int, default=50)
        parser.add_argument("--max_len", type=int, default=120)
        parser.add_argument("--l2_coef", type=float, default=0.001)
        parser.add_argument("--vq_coef", type=float, default=1.0)
    
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
        )

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data = batched_data
        
        # encoding
        codes = self.encoder(batched_input_data)
        if self.hparams.vq:
            codes, _, vq_loss = self.vq_layer(codes)
            loss += self.hparams.vq_coef * vq_loss
            statistics["loss/vq"] = vq_loss
            
        else:
            l2_loss = torch.norm(codes, p=2, dim=1).mean()
            loss += self.hparams.l2_coef * l2_loss
            statistics["loss/l2"] = l2_loss

        # decoding
        logits = self.decoder(batched_target_data, codes)
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)
        elem_acc, seq_acc = compute_sequence_accuracy(logits, batched_target_data)
        loss += recon_loss
        statistics["loss/recon"] = loss
        statistics["acc/elem"] = elem_acc
        statistics["acc/seq"] = seq_acc
        
        
        if self.training and self.hparams.check_sample_quality_freq > 0:
            if (self.global_step + 1) % self.hparams.check_sample_quality_freq == 0:
                target_smiles_list = [
                    self.tokenizer.decode(data).replace(" ", "") for data in batched_target_data.tolist()
                    ]
                target_smiles_list = list(map(canonicalize, target_smiles_list))
                
                with torch.no_grad():
                    batched_recon_data = self.decoder.sample(codes, argmax=True, max_len=self.hparams.max_len)
                
                recon_smiles_list = [
                    self.tokenizer.decode(data).replace(" ", "") for data in batched_recon_data.tolist()
                    ]
                recon_smiles_list = list(map(canonicalize, recon_smiles_list))

                num_valid = len([smi for smi in recon_smiles_list if smi is not None])
                num_correct = len([smi0 for smi0, smi1 in zip(recon_smiles_list, target_smiles_list) if smi0 == smi1])

                statistics["sample/valid_ratio"] = float(num_valid) / len(target_smiles_list)
                statistics["sample/correct_ratio"] = float(num_correct) / len(target_smiles_list)

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        if self.hparams.vq:
            fig = plt.figure()
            val = torch.sort(self.vq_layer.cluster_size, descending=True)[0].cpu().numpy()
            val /= val.sum()
            plt.bar(np.arange(self.hparams.vq_num_vocabs), val)
            self.logger.experiment.log_image('cluster_size', fig)
            plt.close(fig)


    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("validation/loss/total", loss, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]