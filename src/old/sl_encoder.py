import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.pyg.collate import collate_pyg_data_list
from data.util import ZipDataset


def collate_data_list(data_list):
    pyg_data_list, score_list = zip(*data_list)
    score_list = [score[0] for score in score_list]
    return (
        collate_pyg_data_list(pyg_data_list),
        torch.stack(score_list, dim=0),
    )


class SupervisedLearningEncoderModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super(SupervisedLearningEncoderModel, self).__init__()
        self.backbone = backbone
        self.batch_size = hparams.sl_batch_size
        self.num_workers = hparams.sl_num_workers
        self.coef_maxpred = hparams.sl_coef_maxpred

        self.train_dataset = ZipDataset(
            self.backbone.train_pyg_dataset, self.backbone.train_score_dataset
        )
        self.val_dataset = ZipDataset(
            self.backbone.val_pyg_dataset, self.backbone.val_score_dataset
        )

        self.save_hyperparameters(hparams)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--sl_batch_size", type=int, default=128)
        parser.add_argument("--sl_num_workers", type=int, default=8)
        parser.add_argument("--sl_coef_maxpred", type=float, default=0.1)
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
        batched_pyg_data, scores = batched_data
        code = self.backbone.encoder(batched_pyg_data)
        score_preds = self.backbone.score_predictor(code)

        loss_mse = torch.nn.functional.mse_loss(score_preds, scores)
        loss_maxpred = torch.norm(self.backbone.score_predictor.weight, p=2, dim=1)
        loss_total = loss_mse + self.coef_maxpred * loss_maxpred
        return loss_total, {"loss/mse": loss_mse, "loss/maxpred": loss_maxpred}

    def training_step(self, batched_data, batch_idx):
        self.backbone.train()
        loss_total, statistics = self.shared_step(batched_data, batch_idx)
        statistics["stat/maxpred"] = (
            self.backbone.score_mean + self.backbone.score_std * statistics["loss/maxpred"]
        )

        self.log("sl_encoder/train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"sl_encoder/train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        self.backbone.eval()
        with torch.no_grad():
            loss_total, statistics = self.shared_step(batched_data, batch_idx)

        self.log("sl_encoder/validation/loss/total", loss_total, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"sl_encoder/validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        params = list(self.backbone.encoder.parameters())
        params += list(self.backbone.score_predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
