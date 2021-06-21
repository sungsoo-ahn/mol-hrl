import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from net.gnn import GnnEncoder
from data.pyg.dataset import PyGDataset
from data.pyg.handler import PyGHandler
from data.pyg.collate import collate_pyg_data_list
from data.util import ZipDataset, load_split_smiles_list
from util.molecule.scoring.factory import get_scoring_func

class RepresentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super(RepresentationDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_smiles_list, vali_smiles_list = load_split_smiles_list(self.data_dir)
        train_scores = torch.tensor(get_scoring_func("penalized_logp")(train_smiles_list, num_jobs=8))
        vali_scores = torch.tensor(get_scoring_func("penalized_logp")(vali_smiles_list, num_jobs=8))
        print(train_scores)
        assert False
        self.pyg_handler = PyGHandler()
        self.train_smiles_list = train_smiles_list
        self.train_dataset = ZipDataset(
            PyGDataset(train_smiles_list, self.pyg_handler),
            TensorDataset(train_scores),
        )
        self.vali_smiles_list = vali_smiles_list
        self.vali_dataset = ZipDataset(
            PyGDataset(vali_smiles_list, self.pyg_handler),
            TensorDataset(vali_scores),
        )
    
    def collate_data_list(self, data_list):
        pyg_data_list, score_list = zip(*data_list)
        return (
            collate_pyg_data_list(pyg_data_list),
            torch.stack(score_list, dim=0),
        )

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("data")
        group.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        group.add_argument("--batch_size", type=int, default=256)
        group.add_argument("--num_workers", type=int, default=8)
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_data_list,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vali_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_data_list,
            num_workers=self.num_workers,
        )

class RepresentationLearningModel(pl.LightningModule):
    def __init__(
        self,
        encoder_num_layer,
        encoder_emb_dim,
        code_dim,
    ):
        super(RepresentationLearningModel, self).__init__()
        self.encoder = GnnEncoder(num_layer=encoder_num_layer, emb_dim=encoder_emb_dim)
        self.predictor = torch.nn.Linear(code_dim, 1)
        self.save_hyperparameters()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("imitation")
        group.add_argument("--encoder_num_layer", type=int, default=5)
        group.add_argument("--encoder_emb_dim", type=int, default=256)
        group.add_argument("--code_dim", type=int, default=32)
        return parser

    def training_step(self, batched_data, batch_idx):
        batched_pyg_data, scores = batched_data
        out = self.encoder(batched_pyg_data)
        pred = self.predictor(out)
        loss = torch.nn.functional.mse_loss(pred, scores)

        self.log("train/loss/total", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batched_data, batch_idx):
        batched_pyg_data, scores = batched_data
        with torch.no_grad():
            out = self.encoder(batched_pyg_data)
            pred = self.predictor(out)
            loss = torch.nn.functional.mse_loss(pred, scores)

        self.log("validation/loss/total", loss, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        return [optimizer]
