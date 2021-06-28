from tqdm import tqdm
import torch
import pytorch_lightning as pl

from data.pyg.dataset import PyGDataset
from data.sequence.dataset import SequenceDataset
from data.sequence.util import create_tokenizer_and_vocabulary_from_dir
from data.smiles.util import load_split_smiles_list
from data.score.factory import get_scoring_func

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder


class BackBoneModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BackBoneModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.tokenizer, self.vocabulary = create_tokenizer_and_vocabulary_from_dir(hparams.data_dir)
        self.encoder = GnnEncoder(
            num_layer=hparams.encoder_num_layer,
            emb_dim=hparams.encoder_emb_dim,
            code_dim=hparams.code_dim,
        )
        self.decoder = RnnDecoder(
            num_layers=hparams.decoder_num_layers,
            input_dim=len(self.vocabulary),
            output_dim=len(self.vocabulary),
            hidden_dim=hparams.decoder_hidden_dim,
            code_dim=hparams.code_dim,
        )
        self.score_predictor = torch.nn.Linear(hparams.code_dim, 1)
        self.elem_score_func, self.parallel_score_func = get_scoring_func(
            hparams.score_func_name, num_workers=32
        )

        # setup datasets
        (
            (self.train_smiles_list, self.val_smiles_list),
            (self.train_score_list, self.val_score_list),
        ) = load_split_smiles_list(hparams.data_dir, score_func_name=hparams.score_func_name)

        self.train_sequence_dataset = SequenceDataset(
            self.train_smiles_list, self.tokenizer, self.vocabulary
        )
        self.val_sequence_dataset = SequenceDataset(
            self.val_smiles_list, self.tokenizer, self.vocabulary
        )

        self.train_pyg_dataset = PyGDataset(self.train_smiles_list)
        self.val_pyg_dataset = PyGDataset(self.val_smiles_list)

        train_scores = torch.tensor(self.train_score_list).unsqueeze(1)
        self.score_mean = train_scores.mean()
        self.score_std = train_scores.std()
        train_scores = (train_scores - self.score_mean) / self.score_std
        self.train_score_dataset = torch.utils.data.TensorDataset(train_scores)
        val_scores = torch.tensor(self.val_score_list).unsqueeze(1)
        val_scores = (val_scores - self.score_mean) / self.score_std
        self.val_score_dataset = torch.utils.data.TensorDataset(val_scores)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        parser.add_argument("--code_dim", type=int, default=32)
        parser.add_argument("--encoder_num_layer", type=int, default=5)
        parser.add_argument("--encoder_emb_dim", type=int, default=256)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--score_func_name", type=str, default="penalized_logp")
        return parser
