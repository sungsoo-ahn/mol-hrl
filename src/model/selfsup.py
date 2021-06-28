import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.pyg.collate import collate_pyg_data_list


class SelfSupervisedModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super(SelfSupervisedModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.batch_size = hparams.selfsup_batch_size
        self.num_workers = hparams.selfsup_num_workers
        self.projector = torch.nn.Linear(hparams.code_dim, hparams.code_dim)

        self.train_dataset = self.backbone.train_pyg_dataset
        self.val_dataset = self.backbone.val_pyg_dataset

    @staticmethod
    def add_args(parser):
        parser.add_argument("--selfsup_batch_size", type=int, default=256)
        parser.add_argument("--selfsup_num_workers", type=int, default=8)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_pyg_data_list,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_pyg_data_list,
            num_workers=self.num_workers,
        )

    def shared_step(self, batched_data, batch_idx):
        graph_representation, node_representation = self.backbone.encoder(batched_data)
        h0 = F.normalize(self.projector(node_representation), p=2, dim=1)
        h1 = F.normalize(graph_representation, p=2, dim=1)
        logits = torch.matmul(h0, h1.T)
        loss_total = loss_selfsup = F.cross_entropy(logits, batched_data.batch)
        acc_selfsup = (logits.argmax(dim=1) == batched_data.batch).float().mean()

        return loss_total, {"loss/selfsup": loss_selfsup, "acc/selfsup": acc_selfsup,}

    def training_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log("selfsup/train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"selfsup/train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log(
            "selfsup/validation/loss/total", loss_total, on_step=False, logger=True
        )
        for key, val in statistics.items():
            self.log(f"selfsup/validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        params = list(self.backbone.encoder.parameters())
        params += list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
