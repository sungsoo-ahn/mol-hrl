import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from data.pyg.dataset import PyGDataset
from data.pyg.collate import collate_pyg_data_list
from data.score.dataset import ScoresDataset
from data.util import ZipDataset, load_raw_data

def collate_data_list(data_list):
    pyg_data_list, score_data_list = zip(*data_list)
    return collate_pyg_data_list(pyg_data_list), torch.stack(score_data_list, dim=0)

class ScorePredictorModel(pl.LightningModule):
    def __init__(self, backbone, hparams):
        super(ScorePredictorModel, self).__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.setup_datasets(hparams)
        self.setup_models(hparams)

    def setup_datasets(self, hparams):
        smiles_list, scores_list, split_idxs = load_raw_data(
            self.backbone.hparams.data_dir, 
            self.backbone.hparams.score_func_names, 
            self.backbone.hparams.train_ratio, 
            self.backbone.hparams.label_ratio,
        )
        self.datasets = dict()
        self.datasets["full/smiles"] = smiles_list
        self.datasets["full/pyg"] = PyGDataset(smiles_list)
        self.datasets["full/score"] = ScoresDataset(scores_list)
        for split_key in ["train_labeled", "val"]:
            for data_key in ["smiles", "pyg", "score"]:
                self.datasets[f"{split_key}/{data_key}"] = Subset(
                    self.datasets[f"full/{data_key}"], split_idxs[split_key]
                )
        
        self.train_dataset = ZipDataset(
            [self.datasets["train_labeled/pyg"], self.datasets["train_labeled/score"]]
        )
        self.val_dataset = ZipDataset(
            [self.datasets["val/pyg"], self.datasets["val/score"]]
        )

        self.batch_size = hparams.score_predictor_batch_size
        self.num_workers = hparams.score_predictor_num_workers
    
    def setup_models(self, hparams):
        self.score_predictors = torch.nn.ModuleDict(
                {
                    score_func_name: torch.nn.Linear(self.backbone.hparams.code_dim, 1) for 
                    score_func_name in hparams.score_func_names
                }
            )
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--score_predictor_batch_size", type=int, default=256)
        parser.add_argument("--score_predictor_num_workers", type=int, default=24)
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

    def training_step(self, batched_data, batch_idx):
        self.backbone.eval()
        loss_total, statistics = self.shared_step(batched_data)

        self.log("train/loss/total", loss_total, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss_total

    def validation_step(self, batched_data, batch_idx):
        loss_total, statistics = self.shared_step(batched_data)

        self.log("validation/loss/total", loss_total, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)

        return loss_total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score_predictors.parameters(), lr=1e-3)
        return [optimizer]

    def shared_step(self, batched_data):
        batched_pyg_data, scores = batched_data
        codes = self.backbone.encoder(batched_pyg_data)

        statistics = dict()
        loss = 0.0
        for idx, (score_func_name, score_predictor) in enumerate(self.score_predictors.items()):
            scores_pred = score_predictor(codes)
            mse_loss = F.mse_loss(scores_pred.squeeze(1), scores[:, idx])
            loss += mse_loss

            statistics[f"loss/{score_func_name}/mse"] = mse_loss

        return loss, statistics

    def predict_scores(self, code, score_func_name):
        return self.score_predictors[score_func_name](code)