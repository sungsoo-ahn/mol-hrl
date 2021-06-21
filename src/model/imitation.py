import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sequence.dataset import SequenceDataset
from data.sequence.handler import SequenceHandler
from data.sequence.collate import collate_sequence_data_list
from data.pyg.dataset import PyGDataset
from data.pyg.handler import PyGHandler
from data.pyg.collate import collate_pyg_data_list
from data.util import ZipDataset, load_split_smiles_list

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder

from util.sequence import compute_sequence_accuracy, compute_sequence_cross_entropy


class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super(SmilesDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_smiles_list, vali_smiles_list = load_split_smiles_list(self.data_dir)
        self.sequence_handler = SequenceHandler(self.data_dir)
        self.pyg_handler = PyGHandler()
        self.train_smiles_list = train_smiles_list
        self.train_dataset = ZipDataset(
            SequenceDataset(train_smiles_list, self.sequence_handler),
            PyGDataset(train_smiles_list, self.pyg_handler),
        )
        self.vali_smiles_list = vali_smiles_list
        self.vali_dataset = ZipDataset(
            SequenceDataset(vali_smiles_list, self.sequence_handler),
            PyGDataset(vali_smiles_list, self.pyg_handler),
        )
    
    def collate_data_list(self, data_list):
        sequence_data_list, pyg_data_list = zip(*data_list)
        pad_id = self.sequence_handler.vocabulary.get_pad_id()
        return (
            collate_sequence_data_list(sequence_data_list, pad_id), 
            collate_pyg_data_list(pyg_data_list)
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

class ImitationLearningModel(pl.LightningModule):
    def __init__(
        self,
        encoder_num_layer,
        encoder_emb_dim,
        encoder_load_path,
        encoder_optimize,
        decoder_num_layers,
        decoder_hidden_dim,
        decoder_code_dim,
        data_dir,
    ):
        super(ImitationLearningModel, self).__init__()
        self.encoder = GnnEncoder(num_layer=encoder_num_layer, emb_dim=encoder_emb_dim)
        self.encoder_optimize = encoder_optimize
        if encoder_load_path != "":
            self.encoder.load_state_dict(torch.load(encoder_load_path))

        if not self.encoder_optimize:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.sequence_handler = SequenceHandler(data_dir)
        num_vocabs = len(self.sequence_handler.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=decoder_num_layers,
            input_dim=num_vocabs,
            output_dim=num_vocabs,
            hidden_dim=decoder_hidden_dim,
            code_dim=decoder_code_dim,
        )
        
        self.save_hyperparameters()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("imitation")
        group.add_argument("--encoder_optimize", action="store_true")
        group.add_argument("--encoder_num_layer", type=int, default=5)
        group.add_argument("--encoder_emb_dim", type=int, default=300)
        group.add_argument("--encoder_load_path", type=str, default="")
        group.add_argument("--decoder_num_layers", type=int, default=3)
        group.add_argument("--decoder_hidden_dim", type=int, default=1024)
        group.add_argument("--decoder_code_dim", type=int, default=300)
        return parser

    def training_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data = batched_data
        with torch.no_grad():
            codes = self.encoder(batched_pyg_data)
            codes = torch.nn.functional.normalize(codes, p=2, dim=1)

        logits = self.decoder(batched_sequence_data, codes)
        loss = compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )
        elem_acc, sequence_acc = compute_sequence_accuracy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )

        self.log("train/loss/total", loss, on_step=True, logger=True)
        self.log("train/acc/element", elem_acc, on_step=True, logger=True)
        self.log("train/acc/sequence", sequence_acc, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data = batched_data
        with torch.no_grad():
            codes = self.encoder(batched_pyg_data)
            logits = self.decoder(batched_sequence_data, codes)

        loss = compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )
        elem_acc, sequence_acc = compute_sequence_accuracy(
            logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id()
        )

        self.log("validation/loss/total", loss, logger=True)
        self.log("validation/acc/element", elem_acc, logger=True)
        self.log("validation/acc/sequence", sequence_acc, logger=True)

    def configure_optimizers(self):
        params = list(self.decoder.parameters())
        if self.encoder_optimize:
            params += list(self.encoder.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
