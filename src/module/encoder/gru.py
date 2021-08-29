import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from data.util import load_tokenizer

class GRUEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, code_dim):
        super(GRUEncoder, self).__init__()
        self.tokenizer = load_tokenizer()
        num_vocabs = self.tokenizer.get_vocab_size()

        self.encoder = nn.Embedding(num_vocabs, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * hidden_dim, code_dim)

    def forward(self, batched_sequence_data):
        lengths = torch.sum(batched_sequence_data != self.tokenizer.token_to_id("[PAD]"), dim=1)
        out = self.encoder(batched_sequence_data)
        out = pack_padded_sequence(out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False)
        _, h = self.gru(out, None)
        out = torch.cat([h[-2], h[-1]], 1)
        out = self.decoder(out)

        return out