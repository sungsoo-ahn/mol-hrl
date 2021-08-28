import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PlugLSTM(nn.Module):
    def __init__(self, num_layers, hidden_dim, code_dim, vq_num_vocabs):
        super(PlugLSTM, self).__init__()
        self.code_dim = code_dim
        self.vq_num_vocabs = vq_num_vocabs
        self.encoder = nn.Embedding(vq_num_vocabs+1, hidden_dim)
        self.y_encoder = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(hidden_dim, vq_num_vocabs)
        
    def forward(self, x, y, teacher_forcing_ratio=1.0):
        trg = torch.full((x.size(0), ), self.vq_num_vocabs, device=x.device, dtype=torch.long)
        hidden = None
        y_encoded = self.y_encoder(y.unsqueeze(1))
        logits = []
        for t in range(self.code_dim):
            trg_encoded = self.encoder(trg)
            out = trg_encoded + y_encoded
            out, hidden = self.lstm(out.unsqueeze(1), hidden)
            out = self.decoder(out)

            if random.random() < teacher_forcing_ratio:
                trg = x[:, t]
            else:
                probs = torch.softmax(out.squeeze(1), dim=1)
                distribution = Categorical(probs=probs)
                trg = distribution.sample()

            logits.append(out)

        logits = torch.cat(logits, dim=1)
        return logits
    
    def decode(self, y):
        sample_size = y.size(0)
        sequences = [torch.full((sample_size, 1), self.vq_num_vocabs, dtype=torch.long).to(y.device)]
        hidden = None
        code_encoder_out = self.y_encoder(y)
        for _ in range(self.code_dim):
            out = self.encoder(sequences[-1])
            out = out + code_encoder_out.unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            tth_sequences = distribution.sample()    
            sequences.append(tth_sequences)

        sequences = torch.cat(sequences[1:], dim=1)

        return sequences