import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sequence.collate import collate_sequence_data_list
from data.sequence.dataset import SequenceDataset
from data.score.factory import get_scoring_func
from data.util import ZipDataset



def collate_data_list(data_list):
    pyg_data_list, score_list = zip(*data_list)
    score_list = [score[0] for score in score_list]
    return (
        collate_sequence_data_list(pyg_data_list, pad_id=0),
        torch.stack(score_list, dim=0),
    )


class FinetuneModel(pl.LightningModule):
    def __init__(self, backbone, score_func_name, queries, hparams):
        super(FinetuneModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.queries = queries
        self.batch_size = hparams.finetune_batch_size
        self.num_workers = hparams.finetune_num_workers
        self.num_samples_per_query = hparams.finetune_num_samples_per_query
        
        #
        self.elem_score_func, self.parallel_score_func, self.corrupt_score = get_scoring_func(
            score_func_name, num_workers=8
        )
        
        # label only subset
        k = int(hparams.finetune_label_ratio * len(self.backbone.train_smiles_list))
        train_smiles_list = self.backbone.train_smiles_list[:k]
        train_sequence_dataset = SequenceDataset(
            train_smiles_list, self.backbone.tokenizer, self.backbone.vocabulary
        )

        train_score_list = self.parallel_score_func(train_smiles_list)
        train_scores = torch.tensor(train_score_list).unsqueeze(1)
        train_score_dataset = torch.utils.data.TensorDataset(train_scores)
        self.train_dataset = ZipDataset(train_sequence_dataset, train_score_dataset)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--finetune_batch_size", type=int, default=128)
        parser.add_argument("--finetune_num_workers", type=int, default=8)
        parser.add_argument("--finetune_label_ratio", type=float, default=0.01)
        parser.add_argument("--finetune_num_samples_per_query", type=int, default=1024)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def shared_step(self, batched_data, batch_idx):
        batched_sequence_data, scores = batched_data
        codes = self.backbone.score_embedding(scores)
        logits = self.backbone.decoder(batched_sequence_data, codes)

        loss_total = loss_ce = self.backbone.compute_cross_entropy(logits, batched_sequence_data)
        acc_elem, acc_sequence = self.backbone.compute_accuracy(logits, batched_sequence_data)

        return (
            loss_total,
            {"loss/ce": loss_ce, "acc/elem": acc_elem, "acc/sequence": acc_sequence},
        )

    def training_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log("finetune/train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"finetune/train/{key}", val, on_step=True, logger=True)

        return loss_total

    def training_epoch_end(self, training_step_outputs):
        if (self.current_epoch + 1) % 5 == 0:
            for query in self.queries:
                scores = torch.tensor(query, device=self.device)
                scores = scores.view(1, 1).expand(self.num_samples_per_query, 1)
                codes = self.backbone.score_embedding(scores)
                smiles_list = self.backbone.sample_from_codes(codes)
                
                unique_smiles_list = list(set(smiles_list))
                unique_ratio = float(len(unique_smiles_list)) / len(smiles_list)
                
                smiles_list = [smiles for smiles in smiles_list if len(smiles) > 0]
                scores = self.parallel_score_func(smiles_list)
                valid_scores = torch.tensor(
                    [score for score in scores if score > self.corrupt_score]
                    )
                valid_ratio = (valid_scores.size(0) / self.num_samples_per_query)
                
                if valid_scores.size(0) > 0:
                    mean_score = valid_scores.mean()
                    mae = (valid_scores - query).abs().mean()
                else:
                    valid_ratio, mean_score, mae = 0.0, 0.0, 0.0

                self.log(f"finetune/query{query:.1f}/unique_ratio", unique_ratio)
                self.log(f"finetune/query{query:.1f}/valid_ratio", valid_ratio)
                self.log(f"finetune/query{query:.1f}/mean_score", mean_score)
                self.log(f"finetune/query{query:.1f}/mean_abs_error", mae)
                
    def configure_optimizers(self):
        params = list(self.backbone.decoder.parameters())
        params += list(self.backbone.score_embedding.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
