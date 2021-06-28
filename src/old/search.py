from collections import deque
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder, rnn_sample_large
from data.sequence.handler import SequenceHandler
from data.pyg.handler import PyGHandler
from data.sequence.collate import collate_sequence_data_list
from data.pyg.collate import collate_pyg_data_list
from data.util import load_subsampled_train_smiles_list
from util.molecule.scoring.factory import get_scoring_func
from util.molecule.mol import is_valid_smiles
from util.sequence import compute_sequence_cross_entropy
from tqdm import tqdm


class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, element):
        self.buffer.append(element)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class SourceDataset(IterableDataset):
    def __init__(self, generate_batch) -> None:
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


class SearchModel(pl.LightningModule):
    def __init__(
        self,
        encoder_num_layer,
        encoder_emb_dim,
        encoder_load_path,
        encoder_optimize,
        decoder_num_layers,
        decoder_hidden_dim,
        code_dim,
        decoder_load_path,
        decoder_optimize,
        buffer_capacity,
        data_dir,
        batch_size,
        num_workers,
        num_warmup_samples,
        scoring_name,
        queries_per_epoch,
        sample_batch_size,
        batches_per_epoch,
    ):
        super(SearchModel, self).__init__()
        # Encoder
        self.pyg_handler = PyGHandler()
        self.encoder = GnnEncoder(
            num_layer=encoder_num_layer, emb_dim=encoder_emb_dim, code_dim=code_dim
        )
        if encoder_load_path != "":
            self.encoder.load_state_dict(torch.load(encoder_load_path))
        self.encoder_optimize = encoder_optimize
        if not self.encoder_optimize:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Decoder
        self.sequence_handler = SequenceHandler(data_dir)
        num_vocabs = len(self.sequence_handler.vocabulary)
        self.decoder = RnnDecoder(
            num_layers=decoder_num_layers,
            input_dim=num_vocabs,
            output_dim=num_vocabs,
            hidden_dim=decoder_hidden_dim,
            code_dim=code_dim,
        )
        if decoder_load_path != "":
            self.decoder.load_state_dict(torch.load(decoder_load_path))
        self.decoder_optimize = decoder_optimize
        if not self.decoder_optimize:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.eval()

        # Code
        self.target_code = torch.nn.Parameter(torch.randn(code_dim))
        self.pred_bias = torch.nn.Parameter(torch.tensor(0.0))

        # Buffer and dataloading
        self.buffer = Buffer(buffer_capacity)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_warmup_samples = num_warmup_samples

        # Scoring
        self.score = get_scoring_func(scoring_name)

        # Quering
        self.queries_per_epoch = queries_per_epoch
        self.sample_batch_size = sample_batch_size

        # Optimization
        self.step = 0
        self.batches_per_epoch = batches_per_epoch
        self.num_warmup_samples = num_warmup_samples
        self.data_dir = data_dir

        self.max_score = -1000.0

        self.save_hyperparameters()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("imitation")

        # encoder parameters
        group.add_argument("--encoder_num_layer", type=int, default=5)
        group.add_argument("--encoder_emb_dim", type=int, default=256)
        group.add_argument("--encoder_optimize", action="store_true")
        group.add_argument("--encoder_load_path", type=str, default="")

        # decoder parameters
        group.add_argument("--decoder_num_layers", type=int, default=3)
        group.add_argument("--decoder_hidden_dim", type=int, default=1024)
        group.add_argument("--code_dim", type=int, default=32)
        group.add_argument("--decoder_optimize", action="store_true")
        group.add_argument("--decoder_load_path", type=str, default="")

        # buffer and dataloading parameters
        group.add_argument("--buffer_capacity", type=int, default=1024)
        group.add_argument("--data_dir", type=str, default="../resource/data/zinc/")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--num_warmup_samples", type=int, default=1024)

        # objective parameter
        group.add_argument("--scoring_name", type=str, default="penalized_logp")

        # sampling parameters
        group.add_argument("--queries_per_epoch", type=int, default=500)
        group.add_argument("--sample_batch_size", type=int, default=1024)

        # optimizing parameters
        group.add_argument("--batches_per_epoch", type=int, default=1)
        group.add_argument("--batch_size", type=int, default=256)

        return parser

    def populate(self):
        print("Initializing buffer with dataset...")
        smiles_list = load_subsampled_train_smiles_list(self.data_dir, k=self.num_warmup_samples)
        for smiles in tqdm(smiles_list):
            sequence = self.sequence_handler.sequence_from_string(smiles)
            sequence_data = (sequence, torch.tensor(sequence.size(0)))
            pyg_data = self.pyg_handler.pyg_from_string(smiles)

            score = torch.tensor(self.score([smiles], jobs=0)[0])
            self.buffer.append((sequence_data, pyg_data, score))

    def train_dataloader(self):
        self.populate()
        self.dataset = SourceDataset(self.train_batch)
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_data_list,
            num_workers=self.num_workers,
        )

    def collate_data_list(self, data_list):
        sequence_data_list, pyg_data_list, score_list = zip(*data_list)
        return (
            collate_sequence_data_list(
                sequence_data_list, pad_id=self.sequence_handler.vocabulary.get_pad_id()
            ),
            collate_pyg_data_list(pyg_data_list),
            torch.stack(score_list, dim=0),
        )

    def train_batch(self):
        while True:
            self.step += 1
            data_list = self.buffer.sample(self.batch_size)
            for data in data_list:
                yield data

            if self.step % self.batches_per_epoch == 0:
                break

    def training_epoch_end(self, training_step_outputs):
        sequence_data_list, pyg_data_list, scores = self.sample_queries()
        # for sequence_data, pyg_data, score in zip(sequence_data_list, pyg_data_list, scores):
        #    self.buffer.append((sequence_data, pyg_data, score))

    def sample_queries(self):
        self.decoder.eval()
        smiles_list, sequence_data_list, pyg_data_list, scores = [], [], [], []
        target_code = self.target_code.unsqueeze(0).expand(self.sample_batch_size, -1)
        target_code = torch.nn.functional.normalize(target_code, p=2, dim=1)

        num_samples = 0

        while len(scores) < self.queries_per_epoch:
            with torch.no_grad():
                sequences, lengths, _ = self.decoder.sample(
                    target_code,
                    self.sequence_handler.vocabulary.get_start_id(),
                    self.sequence_handler.vocabulary.get_end_id(),
                    self.sequence_handler.vocabulary.get_max_length(),
                )
            batch_smiles_list = self.sequence_handler.strings_from_sequences(sequences, lengths)

            for smiles in batch_smiles_list:
                num_samples += 1
                if not is_valid_smiles(smiles):
                    continue

                score = torch.tensor(self.score([smiles], jobs=0)[0])
                if score.item() < -500:
                    continue

                sequence = self.sequence_handler.sequence_from_string(smiles)
                sequence_data = (sequence, torch.tensor(sequence.size(0)))
                pyg_data = self.pyg_handler.pyg_from_string(smiles)

                smiles_list.append(smiles)
                sequence_data_list.append(sequence_data)
                pyg_data_list.append(pyg_data)
                scores.append(score)

                if not len(scores) < self.queries_per_epoch:
                    break

        num_valid_samples = len(smiles_list)

        self.log("sample/max_score", torch.tensor(scores).max(), logger=True, prog_bar=True)
        self.log(
            "sample/valid_ratio", num_valid_samples / num_samples, logger=True, prog_bar=True,
        )

        return sequence_data_list, pyg_data_list, scores

    def training_step(self, batched_data, batch_idx):
        batched_sequence_data, batched_pyg_data, scores = batched_data
        codes = self.encoder(batched_pyg_data)
        codes = torch.nn.functional.normalize(codes, p=2, dim=1)
        pred = torch.mm(codes, self.target_code.unsqueeze(1)).squeeze(1) + self.pred_bias
        loss = code_loss = torch.nn.functional.mse_loss(pred, scores)
        loss += 1.0 * torch.norm(codes, p=2, dim=1).mean()

        if self.decoder_optimize:
            self.decoder.train()
            logits = self.decoder(batched_sequence_data, codes)
            decoder_loss = compute_sequence_cross_entropy(
                logits, batched_sequence_data, self.sequence_handler.vocabulary.get_pad_id(),
            )
            loss = loss + decoder_loss

        return loss

    def configure_optimizers(self):
        params = [{"params": [self.target_code, self.pred_bias], "lr": 1e-2}]
        if self.decoder_optimize:
            params += [{"params": list(self.decoder.parameters())}]
        if self.encoder_optimize:
            params += [{"params": list(self.encoder.parameters())}]

        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
