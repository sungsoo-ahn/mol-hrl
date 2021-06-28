from joblib import Parallel, delayed
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from net.gnn import GnnEncoder
from net.rnn import RnnDecoder
from data.sequence.handler import SequenceHandler
from data.pyg.handler import PyGHandler, pyg_from_string
from data.pyg.collate import collate_pyg_data_list
from util.molecule.mol import is_valid_smiles


def maybe_get_pyg_data(smiles):
    if not is_valid_smiles(smiles):
        return None

    return pyg_from_string(smiles)


class ReinforceModel(pl.LightningModule):
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
        data_dir,
        batch_size,
        batches_per_epoch,
        reward_temperature,
    ):
        super(ReinforceModel, self).__init__()
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

        # Code
        self.code_dim = code_dim
        self.batch_size = batch_size

        # Optimization
        self.step = 0
        self.batches_per_epoch = batches_per_epoch
        self.reward_temperature = reward_temperature

        self.pool = Parallel(n_jobs=32)

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
        group.add_argument("--decoder_load_path", type=str, default="")
        group.add_argument("--data_dir", type=str, default="../resource/data/zinc/")

        # optimizing parameters
        group.add_argument("--batches_per_epoch", type=int, default=1000)
        group.add_argument("--batch_size", type=int, default=256)
        group.add_argument("--reward_temperature", type=float, default=1.0)

        return parser

    def train_dataloader(self):
        self.dataset = torch.zeros(self.batches_per_epoch, self.code_dim)
        return DataLoader(dataset=self.dataset, batch_size=1)

    def training_step(self, batched_data, batch_idx):
        codes = torch.randn((self.batch_size, self.code_dim), device=self.device)
        codes = torch.nn.functional.normalize(codes, p=2, dim=1)
        sequences, lengths, log_probs = self.decoder.sample(
            codes,
            self.sequence_handler.vocabulary.get_start_id(),
            self.sequence_handler.vocabulary.get_end_id(),
            self.sequence_handler.vocabulary.get_max_length(),
        )

        string_list = self.sequence_handler.strings_from_sequences(sequences, lengths)

        # Filter out smiles and
        pyg_data_list = [maybe_get_pyg_data(string) for string in string_list]
        valid_idxs, valid_pyg_data_list = map(
            list,
            zip(
                *[
                    (idx, pyg_data)
                    for idx, pyg_data in enumerate(pyg_data_list)
                    if pyg_data is not None
                ]
            ),
        )

        # compute actual code
        batched_pyg_data = collate_pyg_data_list(valid_pyg_data_list)
        batched_pyg_data = batched_pyg_data.to(self.device)
        with torch.no_grad():
            relabeled_codes = self.encoder(batched_pyg_data)

        # compute reward
        code_cossim = torch.bmm(
            codes[valid_idxs, :].unsqueeze(1), relabeled_codes.unsqueeze(2)
        ).view(-1)
        reward = torch.zeros(self.batch_size, device=self.device)
        reward[valid_idxs] = torch.exp(self.reward_temperature * code_cossim)

        # compute loss
        loss = -(reward * log_probs).mean()

        self.log(
            "sample/valid_smiles_ratio", len(valid_idxs) / self.batch_size, prog_bar=True,
        )
        self.log("train/loss/total", loss, on_step=True, logger=True)
        self.log("train/stat/reward", reward.mean(), on_step=True, logger=True)
        self.log("train/stat/code_cossim", code_cossim.mean(), on_step=True, logger=True)

        return loss

    def configure_optimizers(self):
        params = list(self.decoder.parameters())
        if self.encoder_optimize:
            params += list(self.encoder.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-3)
        return [optimizer]
