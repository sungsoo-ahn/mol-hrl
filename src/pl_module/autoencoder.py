from argparse import Namespace
from re import S
from data.util import load_tokenizer

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#from module.factory import load_encoder, load_decoder
from module.decoder.lstm import LSTMDecoder
from module.encoder.gnn import GNNEncoder
from module.vq_layer import FlattenedVectorQuantizeLayer
from pl_module.util import compute_sequence_cross_entropy, compute_sequence_accuracy
from data.factory import load_dataset, load_collate
from data.smiles.util import canonicalize



class AutoEncoderModule(pl.LightningModule):
    ### Initialization
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_layers(hparams)
        self.setup_datasets(hparams)
        
    def setup_layers(self, hparams):
        self.encoder = GNNEncoder(
            num_layers=hparams.encoder_num_layers,
            hidden_dim=hparams.encoder_hidden_dim,
            code_dim=hparams.code_dim
        )
        self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=hparams.code_dim
            )

    def setup_datasets(self, hparams):
        self.tokenizer = load_tokenizer()
        self.train_dataset = load_dataset(hparams.dataset_name, hparams.task, "train")
        self.val_dataset = load_dataset("graph2seq", hparams.task, "valid")
        self.collate = load_collate(hparams.dataset_name)

    @staticmethod
    def add_args(parser):
        # data
        parser.add_argument("--task", type=str, default="zinc")
        parser.add_argument("--dataset_name", type=str, default="graph2seq")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        # model - code
        parser.add_argument("--code_dim", type=int, default=256)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # training
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--l2_coef", type=float, default=0.001)
        parser.add_argument("--check_sample_quality_freq", type=int, default=50)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    ### Dataloaders and optimizers
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data = batched_data
        
        # encoding
        codes = self.encoder(batched_input_data)
        l2_loss = torch.norm(codes, p=2, dim=1).mean()
        loss += self.hparams.l2_coef * l2_loss
        statistics["loss/l2"] = l2_loss

        # decoding
        logits = self.decoder(batched_target_data, codes)
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)
        elem_acc, seq_acc = compute_sequence_accuracy(logits, batched_target_data)
        loss += recon_loss
        
        statistics["loss/total"] = loss
        statistics["loss/recon"] = recon_loss
        statistics["acc/elem"] = elem_acc
        statistics["acc/seq"] = seq_acc
                
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def sample(self, batched_input_data):
        with torch.no_grad():
            codes = self.encoder(batched_input_data)
            batched_recon_data = self.decoder.sample(codes, argmax=True, max_len=self.hparams.sample_max_len)
        
        return batched_recon_data
        

    def check_sample_quality(self, batched_data):
        statistics = dict()
        
        batched_input_data, batched_target_data = batched_data
        batched_recon_data = self.sample(batched_input_data)

        def untokenize_and_canonicalize(data):
            return canonicalize(self.tokenizer.decode(data).replace(" ", ""))

        target_smiles_list = list(map(untokenize_and_canonicalize, batched_target_data.tolist()))
        recon_smiles_list = list(map(untokenize_and_canonicalize, batched_recon_data.tolist()))

        num_valid = len([smi for smi in recon_smiles_list if smi is not None])
        num_correct = len([smi0 for smi0, smi1 in zip(recon_smiles_list, target_smiles_list) if smi0 == smi1])
        statistics["sample/valid_ratio"] = float(num_valid) / len(target_smiles_list)
        statistics["sample/correct_ratio"] = float(num_correct) / len(target_smiles_list)
        
        return statistics



class VectorQuantizedAutoEncoderModule(AutoEncoderModule):
    def __init__(self, hparams):
        super(AutoEncoderModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_layers(hparams)
        self.setup_datasets(hparams)
    
    def setup_layers(self, hparams):
        code_dim = hparams.vq_code_dim * hparams.vq_codebook_dim
        self.encoder = GNNEncoder(
            num_layers=hparams.encoder_num_layers,
            hidden_dim=hparams.encoder_hidden_dim,
            code_dim=code_dim
        )
        self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=code_dim
            )
        self.vq_layer = FlattenedVectorQuantizeLayer(hparams.vq_codebook_dim, hparams.vq_num_vocabs)
    
    @staticmethod
    def add_args(parser):
        # data
        parser.add_argument("--task", type=str, default="zinc")
        parser.add_argument("--dataset_name", type=str, default="graph2seq")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        # model - code
        parser.add_argument("--vq_code_dim", type=int, default=64)
        parser.add_argument("--vq_codebook_dim", type=int, default=128)
        parser.add_argument("--vq_num_vocabs", type=int, default=64)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=5)
        parser.add_argument("--encoder_hidden_dim", type=int, default=256)

        # model - decoder
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        # training
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--vq_coef", type=float, default=1.0)
        parser.add_argument("--check_sample_quality_freq", type=int, default=50)
        parser.add_argument("--max_len", type=int, default=120)

        return parser
    
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        batched_input_data, batched_target_data = batched_data
        
        # encoding
        codes = self.encoder(batched_input_data)
        codes, _, vq_loss = self.vq_layer(codes)
        loss += self.hparams.vq_coef * vq_loss

        statistics["loss/vq"] = vq_loss
            
        # decoding
        logits = self.decoder(batched_target_data, codes)
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)
        elem_acc, seq_acc = compute_sequence_accuracy(logits, batched_target_data)
        loss += recon_loss

        statistics["loss/total"] = loss
        statistics["loss/recon"] = recon_loss
        statistics["acc/elem"] = elem_acc
        statistics["acc/seq"] = seq_acc
                
        return loss, statistics

    """
    def training_epoch_end(self, outputs):
        self.plot_cluster_size()

    def plot_cluster_size(self):
        fig = plt.figure()
        val = torch.sort(self.vq_layer.cluster_size, descending=True)[0].cpu().numpy()
        val /= val.sum()
        plt.bar(np.arange(self.hparams.vq_num_vocabs), val)
        self.logger.experiment.log_image('cluster_size', fig)
        plt.close(fig)
    """