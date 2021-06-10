import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical

class RnnGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cond_dim, num_layers):
        super(RnnGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.decoder = torch.nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, x, h, c, lengths):
        out = self.encoder(x)
        out = pack_padded_sequence(out, batch_first=True, lengths=lengths, enforce_sorted=False)
        out, h = self.lstm(out, h)
        out, _ = pad_packed_sequence(out, batch_first=True)

        if c is not None:
            c = c.unsqueeze(1).expand(c.size(0), out.size(1), c.size(1))
            out = torch.cat([out, c], dim=2)

        out = self.decoder(out)
        return out, h

    def sample(self, c, batch_size, max_length, vocab):
        if c is not None:
            c = c.unsqueeze(1)

        x = torch.full((batch_size, 1), vocab.get_start_id(), dtype=torch.long).cuda()
        seq = [x]

        h = None
        terminated = torch.zeros(batch_size, dtype=torch.bool).cuda()
        log_prob = 0.0
        for _ in range(max_length):
            out = self.encoder(x)
            out, h = self.lstm(out, h)
            if c is not None:
                out = torch.cat([out, c], dim=2)

            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            x = distribution.sample()
            log_prob += (~terminated).float() * distribution.log_prob(x).squeeze(1)

            seq.append(x)

            terminated = terminated | (x.squeeze(1) == vocab.get_end_id())
            if terminated.all():
                break

        seq = torch.cat(seq, dim=1)
        
        return seq, log_prob
    
    def max_sample(self, c, batch_size, max_length, vocab):
        if c is not None:
            c = c.unsqueeze(1)

        x = torch.full((batch_size, 1), vocab.get_start_id(), dtype=torch.long).cuda()
        seq = [x]

        h = None
        terminated = torch.zeros(batch_size, dtype=torch.bool).cuda()
        for _ in range(max_length):
            out = self.encoder(x)
            out, h = self.lstm(out, h)
            if c is not None:
                out = torch.cat([out, c], dim=2)

            logit = self.decoder(out)

            x = logit.argmax(dim=2)
            
            seq.append(x)

            terminated = terminated | (x.squeeze(1) == vocab.get_end_id())
            if terminated.all():
                break

        seq = torch.cat(seq, dim=1)
        
        return seq

    def sample_strings(self, c, batch_size, max_length, vocab, tokenizer):
        seq, log_prob = self.sample(c, batch_size, max_length, vocab)

        strings = []
        seqs = seq.cpu().split(1, dim=0)
        for seq in seqs:
            strings.append(tokenizer.untokenize(vocab.decode(seq.squeeze(0).tolist())))
        
        return strings, log_prob

    def max_sample_strings(self, c, batch_size, max_length, vocab, tokenizer):
        seq = self.max_sample(c, batch_size, max_length, vocab)

        strings = []
        seqs = seq.cpu().split(1, dim=0)
        for seq in seqs:
            strings.append(tokenizer.untokenize(vocab.decode(seq.squeeze(0).tolist())))
        
        return strings