from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ae.module import AutoEncoderModule


class LatentRegressorModule(pl.LightningModule):
    def __init__(self, hparams):
        super(LatentRegressorModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        self.ae = AutoEncoderModule.load_from_checkpoint(hparams.ae_checkpoint_path)
        for param in self.ae.parameters():
            param.requires_grad = False

        self.setup_models(hparams)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--ae_checkpoint_path", type=str, default="")
        parser.add_argument("--use_mlp", action="store_true")

    def setup_models(self, hparams):
        if hparams.use_mlp:
            self.regressors = nn.ModuleDict(
                {
                    score_func_name: nn.Sequenctial(
                        nn.Linear(self.ae.hparams.code_dim, self.ae.hparams.code_dim),
                        nn.ReLU(),
                        nn.Linear(self.ae.hparams.code_dim, 1),
                    )
                    for score_func_name in hparams.score_func_names
                }
            )
        else:
            self.regressors = nn.ModuleDict(
                {
                    score_func_name: nn.Linear(self.ae.hparams.code_dim, 1)
                    for score_func_name in hparams.score_func_names
                }
            )

    def training_step(self, batched_data, batch_idx):
        self.ae.eval()
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
        optimizer = torch.optim.Adam(self.regressors.parameters(), lr=1e-3)
        return [optimizer]

    def shared_step(self, batched_data):
        batched_pyg_data, scores = batched_data
        encoder_out = self.ae.compute_encoder_out(batched_pyg_data)
        _, _, codes = self.ae.compute_codes(encoder_out)

        statistics = dict()
        loss = 0.0
        for idx, (score_func_name, score_predictor) in enumerate(self.regressors.items()):
            scores_pred = score_predictor(codes)
            mse_loss = F.mse_loss(scores_pred.squeeze(1), scores[:, idx])
            loss += mse_loss

            statistics[f"loss/{score_func_name}/mse"] = mse_loss.detach().clone()

        return loss, statistics

    def predict_scores(self, code, score_func_name):
        return self.regressors[score_func_name](code)
