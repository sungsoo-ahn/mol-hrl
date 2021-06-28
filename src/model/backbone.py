import torch
from net.rnn import RnnDecoder
from net.gnn import GnnEncoder

import torch
import pytorch_lightning as pl
from torch.autograd.grad_mode import F

from data.sequence.dataset import SequenceDataset
from data.sequence.util import (
    create_tokenizer_and_vocabulary_from_dir,
    string_from_sequence,
)
from data.pyg.dataset import PyGDataset
from data.smiles.util import load_split_smiles_list
from data.score.factory import get_scoring_func
from net.rnn import compute_sequence_accuracy, compute_sequence_cross_entropy

import matplotlib.pyplot as plt


class BackboneModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BackboneModel, self).__init__()
        self.save_hyperparameters(hparams)

        # setup datasets
        self.tokenizer, self.vocabulary = create_tokenizer_and_vocabulary_from_dir(
            hparams.data_dir
        )
        self.elem_score_func, self.parallel_score_func = get_scoring_func(
            hparams.score_func_name, num_workers=8
        )
        (
            (self.train_smiles_list, self.val_smiles_list),
            (self.train_score_list, self.val_score_list),
        ) = load_split_smiles_list(
            hparams.data_dir, score_func_name=hparams.score_func_name,
        )

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
        train_scores = torch.tensor(self.train_score_list).unsqueeze(1)
        val_scores = torch.tensor(self.val_score_list).unsqueeze(1)
        self.score_mean = train_scores.mean()
        self.score_std = train_scores.std()
        self.score_max = train_scores.max()
        self.score_min = train_scores.min()
        self.train_score_dataset = torch.utils.data.TensorDataset(
            self.normalize_score(train_scores)
        )
        self.val_score_dataset = torch.utils.data.TensorDataset(
            self.normalize_score(val_scores)
        )

        # setup evaluation parameters
        self.min_scorecond = hparams.min_scorecond
        self.max_scorecond = hparams.max_scorecond
        self.num_scoreconds = hparams.num_scoreconds
        self.sampling_size = hparams.sampling_size

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
            "--scoreconds",
            type=float,
            nargs="+",
            default=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
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

    def sampling_eval(self):
        scoreconds = torch.linspace(
            self.min_scorecond, self.max_scorecond, self.num_scoreconds
        )

        valid_ratios = torch.zeros(self.num_scoreconds)
        mean_scores = torch.zeros(self.num_scoreconds)
        maes = torch.zeros(self.num_scoreconds)

        for idx, scorecond in enumerate(scoreconds):
            normalized_scorecond = self.normalize_score(scorecond).to(self.device)
            normalized_scorecond = normalized_scorecond.view(1, 1).expand(
                self.sampling_size, 1
            )

            sequences, lengths = self.scorecond_sampling(normalized_scorecond)

            sequences = sequences.cpu().split(1, dim=0)
            lengths = lengths.cpu()
            sequences = [
                sequence[:length] for sequence, length in zip(sequences, lengths)
            ]

            smiles_list = [
                string_from_sequence(sequence, self.tokenizer, self.vocabulary)
                for sequence in sequences
            ]

            scores = self.parallel_score_func(smiles_list)
            valid_scores = torch.tensor([score for score in scores if score > -500])
            if valid_scores.size(0) > 0:
                valid_ratios[idx] = valid_scores.size(0) / self.sampling_size
                mean_scores[idx] = valid_scores.mean()
                maes[idx] = (valid_scores - scorecond).abs().mean()

        return {
            "scoreconds": scoreconds,
            "valid_ratios": valid_ratios,
            "mean_scores": mean_scores,
            "maes": maes,
        }

    def on_validation_epoch_end(self, logger):
        statistics = self.sampling_eval()
        scoreconds = statistics["scoreconds"].numpy()
        mean_scores = statistics["mean_scores"].numpy()
        valid_ratios = statistics["valid_ratios"].numpy()
        maes = statistics["maes"].numpy()

        fig, ax = plt.subplots()
        ax.plot(scoreconds, mean_scores)
        ax.set_xlim([self.min_scorecond - 0.1, self.max_scorecond + 0.1])
        ax.set_ylim([self.min_scorecond - 0.1, self.max_scorecond + 0.1])
        ax.grid()
        logger.experiment.log_image("scorecond_vs_score", fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(scoreconds, valid_ratios)
        ax.set_xlim([self.min_scorecond - 0.1, self.max_scorecond + 0.1])
        ax.set_ylim([0.0 - 0.1, 1.0 + 0.1])
        ax.grid()
        logger.experiment.log_image("scorecond_vs_validratio", fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(scoreconds, maes)
        ax.set_xlim([self.min_scorecond - 0.1, self.max_scorecond + 0.1])
        # ax.set_ylim([0.0-0.1, 1.0+0.1])
        ax.grid()
        logger.experiment.log_image("scorecond_vs_mae", fig)
        plt.close(fig)

        return {"mae": maes.mean()}

    def scorecond_sampling(self, scorecond):
        codes = self.score_embedding(scorecond)
        sequences, lengths, _ = self.decoder.sample(
            codes,
            self.vocabulary.get_start_id(),
            self.vocabulary.get_end_id(),
            self.vocabulary.get_max_length(),
        )

        return sequences, lengths
