import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sequence.collate import collate_sequence_data_list
from data.util import ZipDataset


def collate_data_list(data_list):
    pyg_data_list, score_list = zip(*data_list)
    score_list = [score[0] for score in score_list]
    return (
        collate_sequence_data_list(pyg_data_list, pad_id=0),
        torch.stack(score_list, dim=0),
    )


class ImitationModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super(ImitationModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.batch_size = hparams.imitation_batch_size
        self.num_workers = hparams.imitation_num_workers

        k = int(hparams.imitation_subset_ratio * len(self.backbone.train_smiles_list))
        print(k)
        subset_idxs = random.Random(0).sample(
            range(len(self.backbone.train_smiles_list)), k=k
        )
        train_sequence_dataset = torch.utils.data.Subset(
            self.backbone.train_sequence_dataset, subset_idxs
        )
        train_score_dataset = torch.utils.data.Subset(
            self.backbone.train_score_dataset, subset_idxs
        )

        self.train_dataset = ZipDataset(train_sequence_dataset, train_score_dataset)
        self.val_dataset = ZipDataset(
            self.backbone.val_sequence_dataset, self.backbone.val_score_dataset
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument("--imitation_batch_size", type=int, default=128)
        parser.add_argument("--imitation_num_workers", type=int, default=8)
        parser.add_argument("--imitation_subset_ratio", type=float, default=0.01)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def shared_step(self, batched_data, batch_idx):
        batched_sequence_data, scores = batched_data
        codes = self.backbone.score_embedding(scores)
        logits = self.backbone.decoder(batched_sequence_data, codes)

        loss_total = loss_ce = self.backbone.compute_cross_entropy(
            logits, batched_sequence_data
        )
        acc_elem, acc_sequence = self.backbone.compute_accuracy(
            logits, batched_sequence_data
        )

        return (
            loss_total,
            {"loss/ce": loss_ce, "acc/elem": acc_elem, "acc/sequence": acc_sequence},
        )

    def training_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log("imitation/train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"imitation/train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log(
            "imitation/validation/loss/total", loss_total, on_step=False, logger=True
        )
        for key, val in statistics.items():
            self.log(f"imitation/validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        params = list(self.backbone.decoder.parameters())
        params += list(self.backbone.score_embedding.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]

    def on_validation_epoch_end(self):
        statistics = self.backbone.on_validation_epoch_end(self.logger)
        for key, val in statistics.items():
            self.log(f"imitation/sampling/{key}", val, on_step=False, logger=True)
