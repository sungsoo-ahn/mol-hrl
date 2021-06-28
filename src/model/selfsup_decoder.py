import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sequence.collate import collate_sequence_data_list
from data.pyg.collate import collate_pyg_data_list
from data.util import ZipDataset


def collate_data_list(data_list):
    sequence_data_list, pyg_data_list = zip(*data_list)
    return (
        collate_sequence_data_list(sequence_data_list, pad_id=0),
        collate_pyg_data_list(pyg_data_list),
    )


class SelfSupervisedDecoderModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super(SelfSupervisedDecoderModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.batch_size = hparams.selfsupdecoder_batch_size
        self.num_workers = hparams.selfsupdecoder_num_workers

        self.train_dataset = ZipDataset(
            self.backbone.train_sequence_dataset, self.backbone.train_pyg_dataset
        )
        self.val_dataset = ZipDataset(
            self.backbone.val_sequence_dataset, self.backbone.val_pyg_dataset
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument("--selfsupdecoder_batch_size", type=int, default=128)
        parser.add_argument("--selfsupdecoder_num_workers", type=int, default=8)
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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_data_list,
            num_workers=self.num_workers,
        )

    def shared_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data = batched_data
        codes, _ = self.backbone.encoder(batched_pyg_data)
        logits = self.backbone.decoder(batched_sequence_data, codes)

        loss_total = loss_ce = self.backbone.compute_cross_entropy(logits, batched_sequence_data)
        acc_elem, acc_sequence = self.backbone.compute_accuracy(logits, batched_sequence_data)

        return (
            loss_total,
            {"loss/ce": loss_ce, "acc/elem": acc_elem, "acc/sequence": acc_sequence},
        )

    def training_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log("selfsupdecoder/train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"selfsupdecoder/train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log(
            "selfsupdecoder/validation/loss/total", loss_total, on_step=False, logger=True,
        )
        for key, val in statistics.items():
            self.log(f"selfsupdecoder/validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        params = list(self.backbone.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]