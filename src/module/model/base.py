from argparse import Namespace
import torch
import pytorch_lightning as pl

class BaseAEModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseAEModule, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)    
        self.setup_models(hparams)
    
    def setup_models(self, hparams):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        raise NotImplementedError
        
    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)
        
        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        self.log("validation/loss/total", loss, on_step=False, logger=True)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def shared_step(self, batched_data):
        raise NotImplementedError