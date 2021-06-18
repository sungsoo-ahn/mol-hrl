import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical


class Rnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, seqs, lengths):
        out = self.encoder(seqs)

        out = pack_padded_sequence(
            out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False
        )
        out, _ = self.lstm(out, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        logit = self.decoder(out)

        return logit

    def sample(self, sample_size, start_id, end_id, max_length):
        seqs = [torch.full((sample_size, 1), start_id, dtype=torch.long).cuda()]

        hidden = None
        terminated = torch.zeros(sample_size, dtype=torch.bool).cuda()
        log_probs = 0.0
        lengths = torch.ones(sample_size, dtype=torch.long).cuda()
        for _ in range(max_length):
            out = self.encoder(seqs[-1])
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            tth_seqs = distribution.sample()

            log_probs += (~terminated).float() * distribution.log_prob(
                tth_seqs
            ).squeeze(1)

            seqs.append(tth_seqs)

            lengths[~terminated] += 1
            terminated = terminated | (tth_seqs.squeeze(1) == end_id)

            if terminated.all():
                break

        seqs = torch.cat(seqs, dim=1)

        return seqs, lengths, log_probs


class ConditionalRnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cond_dim, num_layers):
        super(ConditionalRnn, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.cond_encoder = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, seqs, cond, lengths):
        out = self.encoder(seqs)

        cond = cond.unsqueeze(1).expand(-1, seqs.size(1), -1)
        out = out + self.cond_encoder(cond)

        out = pack_padded_sequence(
            out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False
        )
        out, _ = self.lstm(out, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        logit = self.decoder(out)
        
        return logit

    def sample(self, sample_size, cond, start_id, end_id, max_length):
        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(sample_size, -1)

        seqs = [torch.full((sample_size, 1), start_id, dtype=torch.long).cuda()]
        hidden = None
        terminated = torch.zeros(sample_size, dtype=torch.bool).cuda()
        log_probs = 0.0
        lengths = torch.ones(sample_size, dtype=torch.long).cuda()

        for _ in range(max_length):
            out = self.encoder(seqs[-1])
            out = out + self.cond_encoder(cond)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            tth_seqs = distribution.sample()

            log_probs += (~terminated).float() * distribution.log_prob(
                tth_seqs
            ).squeeze(1)

            seqs.append(tth_seqs)

            lengths[~terminated] += 1
            terminated = terminated | (tth_seqs.squeeze(1) == end_id)

            if terminated.all():
                break

        seqs = torch.cat(seqs, dim=1)

        return seqs, lengths, log_probs