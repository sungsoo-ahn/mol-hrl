import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from data.sequence.util import (
    load_tokenizer,
    load_vocabulary,
    smiles2sequence,
)
from data.sequence.dataset import SequenceDataset


class SequenceEncoder(nn.Module):
    def __init__(self, hparams):
        super(SequenceEncoder, self).__init__()
        self.hparams = hparams
        self.vocabulary = load_vocabulary(hparams.data_dir)
        self.tokenizer = load_tokenizer(hparams.data_dir)
        num_vocabs = len(self.vocabulary)

        self.encoder = nn.Embedding(num_vocabs, hparams.sequence_encoder_hidden_dim)
        self.lstm = nn.LSTM(
            hparams.sequence_encoder_hidden_dim,
            hparams.sequence_encoder_hidden_dim,
            batch_first=True,
            num_layers=hparams.sequence_encoder_num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * hparams.sequence_encoder_hidden_dim, hparams.code_dim)

    def forward(self, batched_sequence_data):
        sequences, lengths = batched_sequence_data
        out = self.encoder(sequences)
        out = pack_padded_sequence(
            out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False
        )
        _, (h, _) = self.lstm(out, None)
        out = torch.cat([h[-2], h[-1]], 1)
        out = self.decoder(out)

        return out

    def encode_smiles(self, smiles_list, device):
        sequences = [
            smiles2sequence(smiles, self.tokenizer, self.vocabulary) for smiles in smiles_list
        ]
        lengths = [torch.tensor(sequence.size(0)) for sequence in sequences]
        data_list = list(zip(sequences, lengths))
        batched_sequence_data = SequenceDataset.collate_fn(data_list)
        batched_sequence_data = [item.to(device) for item in batched_sequence_data]
        return self(batched_sequence_data)
