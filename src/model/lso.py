from data.score.factory import get_scoring_func
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

class LatentSpaceOptimizationModel(pl.LightningModule):
    def __init__(self, backbone, score_predictor, scoring_func_name, hparams):
        super(LatentSpaceOptimizationModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.score_predictor = score_predictor
        for param in self.score_predictor.parameters():
            param.requires_grad = False

        
        self.scoring_func_name = scoring_func_name

        # For scoring at evaluation
        _, self.score_func, self.corrupt_score  = get_scoring_func(scoring_func_name)
        

        # For code optimization
        self.setup_codes()
        self.num_steps_per_epoch = hparams.lso_num_steps_per_epoch
        self.code_lr = hparams.lso_lr

    def train_dataloader(self):
        return DataLoader(torch.arange(self.num_steps_per_epoch))

    @staticmethod
    def add_args(parser):
        parser.add_argument("--lso_num_codes", type=int, default=1024)
        parser.add_argument("--lso_num_steps_per_epoch", type=int, default=10)
        parser.add_argument("--lso_lr", type=float, default=1e-2)
        
    def setup_codes(self):
        smiles_list = self.score_predictor.datasets["train_labeled/smiles"]
        score_list = self.score_func(smiles_list)
        smiles_list = [smiles for _, smiles in sorted(zip(score_list, smiles_list), reverse=True)]
        
        init_codes = self.backbone.encode(smiles_list)
        self.codes = torch.nn.Parameter(init_codes)

    def training_step(self, batched_data, batch_idx):
        self.backbone.eval()
        self.score_predictor.eval()

        pred_scores = self.score_predictor.predict_scores(self.codes, self.scoring_func_name)
        loss = -pred_scores.sum()
        self.log(f"lso/{self.scoring_func_name}/loss/total", loss, on_step=True, logger=True)

        statistics = {
            "pred/max": pred_scores.max(),
            "pred/mean": pred_scores.mean(),
        }
        for key, val in statistics.items():
            self.log(f"lso/{self.scoring_func_name}/{key}", val, on_step=True, logger=True)        

        return loss

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            smiles_list = self.backbone.decode(self.codes)
        
        scores = torch.FloatTensor(self.score_func(smiles_list))
        clean_scores = scores[scores > self.corrupt_score + 1e-3]
        clean_ratio = clean_scores.size(0) / scores.size(0)

        statistics = dict()
        statistics["clean_ratio"] = 0.0
        if clean_ratio > 0.0: 
            statistics["score/max"] = clean_scores.max()
            statistics["score/mean"] = clean_scores.mean()         
        else:
            statistics["score/max"] = 0.0
            statistics["score/mean"] = 0.0

        for key, val in statistics.items():
            self.log(f"lso/{key}", val, on_step=False, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([self.codes], lr=self.code_lr)
        return [optimizer]