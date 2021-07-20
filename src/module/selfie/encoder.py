import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from data.selfie.vocab import (
    load_selfie_vocabulary, 
    load_selfie_tokenizer, 
    selfie_sequence_from_smiles, 
)
from data.selfie.dataset import SelfieDataset

class SelfiesEncoder(nn.Module):
    def __init__(self, hparams):
        super(SelfiesEncoder, self).__init__()
        self.hparams = hparams
        self.vocabulary = load_selfie_vocabulary(hparams.data_dir)
        self.tokenizer = load_selfie_tokenizer(hparams.data_dir)
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
            selfie_sequence_from_smiles(smiles, self.tokenizer, self.vocabulary) for smiles in smiles_list
        ]
        lengths = [torch.tensor(sequence.size(0)) for sequence in sequences]
        data_list = list(zip(sequences, lengths))
        batched_sequence_data = SelfieDataset.collate_fn(data_list)
        batched_sequence_data = [item.to(device) for item in batched_sequence_data]
        return self(batched_sequence_data)
    
    def get_dataset(self, split):
        return SelfieDataset(
            self.hparams.data_dir, 
            split, 
            self.hparams.input_smiles_transform_type, 
            self.hparams.input_sequence_transform_type,
            )