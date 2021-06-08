import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class SmilesLstm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cond_dim, num_layers):
        super(SmilesLstm, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim + cond_dim, output_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, x, h, c, lengths):
        out = self.encoder(x)
        out = pack_padded_sequence(out, batch_first=True, lengths=lengths, enforce_sorted=False)
        out, h = self.lstm(out, h)
        out, _ = pad_packed_sequence(out, batch_first=True)

        if c is not None:
            c = c.unsqueeze(1).repeat(out.size(1), dim=1)
            out = torch.cat([out, c], dim=1)

        out = self.decoder(out)
        return out, h
