from argparse import Namespace
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from ae.module import AutoEncoderModule
from lso.regressor_module import LatentRegressorModule
from data.smiles.util import load_smiles_list
from data.score.factory import get_scoring_func

class LatentOptimizationModule(pl.LightningModule):
    def __init__(self, hparams):
        super(LatentOptimizationModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.ae = AutoEncoderModule.load_from_checkpoint(hparams.ae_checkpoint_path)
        for param in self.ae.parameters():
            param.requires_grad = False

        self.regressor = LatentRegressorModule.load_from_checkpoint(hparams.lso_checkpoint_path)
        for param in self.regressor.parameters():
            param.requires_grad = False

        # For scoring at evaluation
        _, self.score_func, self.corrupt_score = get_scoring_func(hparams.scoring_func_name)

        # For code optimization
        self.setup_codes()
        self.num_steps_per_epoch = hparams.num_steps_per_epoch
        self.code_lr = hparams.lr

    def train_dataloader(self):
        return DataLoader(torch.arange(self.num_steps_per_epoch))

    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc_small")
        parser.add_argument("--num_opt_codes", type=int, default=1024)
        parser.add_argument("--num_steps_per_epoch", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-2)

        parser.add_argument("--scoring_func_name", type=str, default="penalized_logp")
        parser.add_argument("--ae_checkpoint_path", type=str, default="")
        parser.add_argument("--lso_checkpoint_path", type=str, default="")

    def setup_codes(self):
        smiles_list = load_smiles_list(self.hparams.data_dir, split="train_labeled")
        score_list = self.score_func(smiles_list)
        smiles_list = [smiles for _, smiles in sorted(zip(score_list, smiles_list), reverse=True)]
        smiles_list = smiles_list[:self.hparams.num_opt_codes]

        init_codes = self.ae.encoder.encode_smiles(smiles_list)
        self.codes = torch.nn.Parameter(init_codes)

    def training_step(self, batched_data, batch_idx):
        self.ae.eval()
        self.regressor.eval()

        pred_scores = self.regressor.predict_scores(self.codes, self.hparams.scoring_func_name)
        loss = -pred_scores.sum()
        self.log(
            f"lso/{self.hparams.scoring_func_name}/loss/total", loss, on_step=True, logger=True
            )

        statistics = {
            "pred/max": pred_scores.max(),
            "pred/mean": pred_scores.mean(),
        }
        for key, val in statistics.items():
            self.log(f"lso/{self.hparams.scoring_func_name}/{key}", val, on_step=True, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            smiles_list = self.ae.decoder.decode_smiles(self.codes, deterministic=True)

        scores = torch.FloatTensor(self.score_func(smiles_list))
        clean_scores = scores[scores > self.corrupt_score + 1e-3]
        clean_ratio = clean_scores.size(0) / scores.size(0)

        statistics = dict()
        statistics["clean_ratio"] = clean_ratio
        if clean_ratio > 0.0:
            statistics["score/max"] = clean_scores.max()
            statistics["score/mean"] = clean_scores.mean()
        else:
            statistics["score/max"] = 0.0
            statistics["score/mean"] = 0.0

        for key, val in statistics.items():
            self.log(f"lso/{self.hparams.scoring_func_name}/{key}", val, on_step=False, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([self.codes], lr=self.code_lr)
        return [optimizer]
