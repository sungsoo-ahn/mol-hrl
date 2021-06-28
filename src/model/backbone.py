import torch
from net.rnn import RnnDecoder
from net.gnn import GnnEncoder

import torch
import pytorch_lightning as pl
from torch.autograd.grad_mode import F

from data.sequence.dataset import SequenceDataset
from data.sequence.util import create_tokenizer_and_vocabulary_from_dir, string_from_sequence
from data.pyg.dataset import PyGDataset
from data.smiles.util import load_split_smiles_list
from net.rnn import compute_sequence_accuracy, compute_sequence_cross_entropy

class BackboneModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BackboneModel, self).__init__()
        self.save_hyperparameters(hparams)

        # setup datasets
        self.tokenizer, self.vocabulary = create_tokenizer_and_vocabulary_from_dir(hparams.data_dir)
        self.train_smiles_list, self.val_smiles_list = load_split_smiles_list(hparams.data_dir)

        #
        self.train_sequence_dataset = SequenceDataset(
            self.train_smiles_list, self.tokenizer, self.vocabulary
        )
        self.val_sequence_dataset = SequenceDataset(
            self.val_smiles_list, self.tokenizer, self.vocabulary
        )

        #
        self.train_pyg_dataset = PyGDataset(self.train_smiles_list)
        self.val_pyg_dataset = PyGDataset(self.val_smiles_list)

        #
        self.decoder = RnnDecoder(
            num_layers=hparams.decoder_num_layers,
            input_dim=len(self.vocabulary),
            output_dim=len(self.vocabulary),
            hidden_dim=hparams.decoder_hidden_dim,
            code_dim=hparams.code_dim,
        )
        self.encoder = GnnEncoder(
            num_layer=hparams.encoder_num_layer,
            emb_dim=hparams.encoder_emb_dim,
            code_dim=hparams.code_dim,
        )
        self.score_embedding = torch.nn.Linear(1, hparams.code_dim)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        
        parser.add_argument("--code_dim", type=int, default=32)
        parser.add_argument("--encoder_num_layer", type=int, default=5)
        parser.add_argument("--encoder_emb_dim", type=int, default=256)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        
        parser.add_argument("--score_func_name", type=str, default="penalized_logp")
        parser.add_argument(
            "--scoreconds", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        )
        parser.add_argument("--num_scoreconds", type=int, default=21)
        parser.add_argument("--sampling_size", type=int, default=128)

        return parser

    def normalize_score(self, scores):
        return (scores - self.score_mean) / self.score_std

    def unnormalize_score(self, scores):
        return self.score_std * scores + self.score_mean

    def compute_cross_entropy(self, logits, batched_sequence_data):
        return compute_sequence_cross_entropy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
        )

    def compute_accuracy(self, logits, batched_sequence_data):
        return compute_sequence_accuracy(
            logits, batched_sequence_data, self.vocabulary.get_pad_id()
        )

    def sample_from_codes(self, codes):
        sequences, lengths, _ = self.decoder.sample(
            codes,
            self.vocabulary.get_start_id(),
            self.vocabulary.get_end_id(),
            self.vocabulary.get_max_length(),
        )
        
        sequences = sequences.cpu().split(1, dim=0)
        lengths = lengths.cpu()
        sequences = [sequence[:length] for sequence, length in zip(sequences, lengths)]

        smiles_list = [
            string_from_sequence(sequence, self.tokenizer, self.vocabulary)
            for sequence in sequences
        ]
        return smiles_list