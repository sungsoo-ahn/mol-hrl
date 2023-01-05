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
from module.decoder.transformer import TransformerDecoder
from module.decoder.gnn import GNNDecoder
from module.decoder.mlp import MLPDecoder



from module.encoder.gnn import GNNEncoder
from module.vq_layer import FlattenedVectorQuantizeLayer
from pl_module.util import compute_node_cross_entropy, compute_node_accuracy, compute_edge_cross_entropy
from data.factory import load_dataset, load_collate
from data.smiles.util import canonicalize



class VarAutoEncoderModule(pl.LightningModule):
    ### Initialization
    def __init__(self, hparams):
        super(VarAutoEncoderModule, self).__init__()
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
        self.fc_mu = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        self.fc_logvar = torch.nn.Linear(hparams.code_dim, hparams.code_dim)
        if hparams.decoder == "lstm":
            self.decoder = LSTMDecoder(
            num_layers=hparams.decoder_num_layers, 
            hidden_dim=hparams.decoder_hidden_dim, 
            code_dim=hparams.code_dim
            )
        elif hparams.decoder == "transformer":
            self.decoder = TransformerDecoder(
                num_encoder_layers = hparams.encoder_layers,
                emb_size = hparams.emb_size,
                nhead = hparams.nhead,
                dim_feedforward = hparams.dim_feedforward,
                dropout = hparams.dropout,
                code_dim = hparams.code_dim
            )
        elif hparams.decoder == "gnn":
            self.decoder = GNNDecoder(
                num_layers=hparams.decoder_num_layers_mlp,
                hidden_dim=hparams.decoder_hidden_dim_mlp,
                code_dim=hparams.code_dim
            )
        elif hparams.decoder == "mlp":
            self.decoder = MLPDecoder(
                num_layers=hparams.decoder_num_layers_mlp,
                hidden_dim=hparams.decoder_hidden_dim_mlp,
                code_dim=hparams.code_dim
            )

    def setup_datasets(self, hparams):
        self.tokenizer = load_tokenizer()
        self.train_dataset = load_dataset(hparams.dataset_name, hparams.task, "train")
        #self.val_dataset = load_dataset("graph2seq", hparams.task, "valid")
        self.val_dataset = load_dataset(hparams.dataset_name, hparams.task, "valid")
        self.collate = load_collate(hparams.dataset_name)

    @staticmethod
    def add_args(parser):
        # data
        parser.add_argument("--task", type=str, default="zinc")
        parser.add_argument("--dataset_name", type=str, default="graph2graph")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        # model - code
        #parser.add_argument("--code_dim", type=int, default=512)
        
        # model - encoder
        parser.add_argument("--encoder_num_layers", type=int, default=2)

        #parser.add_argument("--encoder_hidden_dim", type=int, default=512)

        # model - decoder
        #parser.add_argument("--decoder", type=str, default="gnn")

        # model - decoder (for lstm)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)

        # model - decoder (for transformer)
        parser.add_argument("--encoder_layers", type=int, default=3)
        #parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)

        # model - decoder (for mlp)
        parser.add_argument("--decoder_num_layers_mlp", type=int, default=2)
        parser.add_argument("--decoder_hidden_dim_mlp", type=int, default=256)



        # training
        parser.add_argument("--lr", type=float, default=1e-4) # default: 1e-4
        parser.add_argument("--kl_coef", type=float, default=0.000001)
        parser.add_argument("--check_sample_quality_freq", type=int, default=50)
        parser.add_argument("--max_len", type=int, default=120)

        return parser

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False, # should be True
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
        
        #mu = self.fc_mu(codes)
        #logvar = self.fc_logvar(codes)

        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)

        #codes = eps * std + mu
        
        #kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        #l2_loss = torch.norm(codes, p=2, dim=1).mean()
        #loss += self.hparams.kl_coef * kl_loss
        #statistics["loss/kl"] = kl_loss
        
        logits_atom = self.decoder(codes, batched_input_data["batch"])
        atom_recon_loss = compute_node_cross_entropy(logits_atom, batched_target_data)
        atom_elem_acc, atom_mol_acc = compute_node_accuracy(logits_atom, batched_target_data)
        #loss += atom_recon_loss

        edge_recon_loss = 0.0

        edge_targets_sparse = batched_input_data["edge_index"].T
        counts = torch.bincount(batched_input_data["batch"])
        idx = 0
        edge_idx = 0
        edge_targets = torch.zeros(len(counts), max(counts), max(counts)).cuda()
        edge_mask = torch.zeros(len(counts), max(counts), max(counts)).cuda()
        

        h_adjs = torch.zeros(len(counts), max(counts), max(counts), 2 * codes.shape[1]).cuda()
        
        idx = 0
        
        for b in range(len(counts)):
            h_mol = codes[idx : idx + counts[b],:]
            
            h_mol1 = h_mol.repeat(counts[b],1).reshape(counts[b], counts[b], -1)
            h_mol2 = h_mol.repeat_interleave(counts[b], dim=0).reshape(counts[b], counts[b], -1)
            h_mol = torch.cat((h_mol1, h_mol2), -1)
            h_adjs[b,:h_mol.shape[0], :h_mol.shape[1], :2*codes.shape[1]] = h_mol
            idx += counts[b]
        
        tmp1 = h_adjs[:,:,:,:codes.shape[1]]
        tmp2 = h_adjs[:,:,:,codes.shape[1]:]

        logits_edge = torch.einsum('bcde,bcde->bcd', tmp1, tmp2)

        idx = 0
        edge_idx = 0
        for b in range(len(counts)):
            tmp_mask1 = edge_targets_sparse >= idx 
            tmp_mask2 = edge_targets_sparse < idx + counts[b]
            tmp_mask = torch.logical_and(tmp_mask1, tmp_mask2)
            edge_mol_sparse = (edge_targets_sparse[tmp_mask] - idx).reshape(-1,2)

            for (i,bond) in enumerate(edge_mol_sparse):
                #print(bond)
                #print(bond[0])
                edge_targets[b,bond[0],bond[1]] = 0 # should be changed to 1
            edge_mask[b,:counts[b],:counts[b]] = 1.0
        
            edge_mask[b,:counts[b],:counts[b]] = edge_mask[b,:counts[b],:counts[b]] - torch.eye(counts[b].item()).cuda()
            idx += counts[b]
            edge_idx += len(edge_mol_sparse)
        print('dd')
        print(logits_edge)
        edge_recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_edge, edge_targets, reduction='none')
        
        print(edge_recon_loss)
        print(edge_recon_loss * edge_mask)
        
        edge_recon_loss = (edge_recon_loss * edge_mask).mean()

        edge_preds = (logits_edge > 0).long()
        edge_preds[edge_mask == 0.0] = 0

        edge_elem_acc = (edge_preds == edge_targets).float().mean()

        edge_mol_acc = (edge_preds == edge_targets).view(edge_preds.shape[0], -1).all(dim=1).float().mean()
    
        pred_zero = (edge_preds == 0).sum()
        gt_zero = (edge_targets == 0).sum()

        pred_one = (edge_preds == 1).sum()
        gt_one = (edge_targets == 1).sum()
        #print("pred")
        #for i in range(len(edge_preds[0])):
        #    print(edge_preds[0,i,:])
        #print("target")
        #for j in range(len(edge_targets[0])):
        #    print(edge_targets[0,j,:])






        # decoding
        #logits_atom = self.decoder(codes, batched_input_data["batch"])


        #atom_recon_loss, chirality_recon_loss = compute_node_cross_entropy(logits_atom, logits_chirality, batched_target_data)
        #atom_elem_acc, atom_mol_acc, chirality_elem_acc, chirality_mol_acc = compute_node_accuracy(logits_atom, logits_chirality, batched_target_data)

        #edge_recon_loss, edge_elem_acc, edge_mol_acc, pred_zero, gt_zero, pred_one, gt_one = compute_edge_cross_entropy(logits_edge, batched_target_data)
        
        #loss += atom_recon_loss
        loss += edge_recon_loss
        #loss += chirality_recon_loss
        
        statistics["loss/total"] = loss
        statistics["loss/atom_recon"] = atom_recon_loss
        statistics["loss/edge_recon"] = edge_recon_loss
        #statistics["loss/chirality_recon"] = chirality_recon_loss
        statistics["acc/atom_elem"] = atom_elem_acc
        statistics["acc/atom_mol"] = atom_mol_acc
        statistics["acc/edge_elem"] = edge_elem_acc
        statistics["acc/edge_mol"] = edge_mol_acc
        #statistics["acc/chirality_elem"] = chirality_elem_acc
        #statistics["acc/chirality_mol"] = chirality_mol_acc
        statistics["loss/pred_zero"] = pred_zero
        statistics["loss/gt_zero"] = gt_zero
        statistics["loss/pred_one"] = pred_one
        statistics["loss/gt_one"] = gt_one
                
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)#, batch_size=self.hparams.batch_size)
        
        
        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        #quality_statistics = self.check_sample_quality(batched_data)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)#, batch_size=self.hparams.batch_size)
        #for key, val in quality_statistics.items():
        #    self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)

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

