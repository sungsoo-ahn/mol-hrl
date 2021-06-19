import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from data.datamodule import SequencePyGDataModule
from learning.imitation import ImitationLearningModel

if __name__ == "__main__":
    model = ImitationLearningModel.load_from_checkpoint(
        "../resource/log/neptune-imitation/HRL-300/checkpoints/epoch=0-step=18.ckpt"
    )
