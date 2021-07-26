import torch
import torch.nn.functional as F

from module.autoencoder.base import BaseAutoEncoder
from data.util import ZipDataset
from data.sequence.dataset import SequenceDataset 
from data.score.dataset import ScoreDataset
from data.graph.dataset import GraphDataset


class SupervisedAutoEncoder(BaseAutoEncoder):
    def __init__(self, hparams):
        super(SupervisedAutoEncoder, self).__init__(hparams)
        self.regressor = torch.nn.Linear(hparams.code_dim, 1)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--supervised_loss_coef", type=float, default=1.0)

    def update_encoder_loss(self, batched_input_data, loss=0.0, statistics=dict()):
        batched_input_data0, target = batched_input_data
        codes = self.encoder(batched_input_data0)
        pred = self.regressor(codes)
        mse_loss = F.mse_loss(pred, target)
        loss += self.hparams.supervised_loss_coef * mse_loss
        statistics["loss/mse"] = mse_loss

        return codes, loss, statistics

    def get_input_dataset(self, split):
        return ZipDataset(
            GraphDataset(self.hparams.data_dir, split),
            ScoreDataset(self.hparams.data_dir, ["penalized_logp"], split),
        )

    @staticmethod
    def collate(data_list):
        input_data_list, target_data_list = zip(*data_list)
        graph_data_list, score_data_list = zip(*input_data_list)
        return (
            (GraphDataset.collate(graph_data_list), ScoreDataset.collate(score_data_list)), 
            SequenceDataset.collate(target_data_list)
        )