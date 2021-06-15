import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical

def compute_rnn_accuracy(logits, y, batch_size):
    y_pred = torch.argmax(logits, dim=-1)
    correct = (y_pred == y)
    correct[y == 0] = True
    elem_acc = correct[y != 0].float().mean()
    seq_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, seq_acc

def compute_rnn_ce(logits, y_seq, lengths):
    logits = logits.view(-1, logits.size(-1))
    y_seq = y_seq.reshape(-1)
    
    loss = torch.nn.functional.cross_entropy(logits, y_seq, reduction="sum", ignore_index=0)
    loss /= torch.sum(lengths - 1)

    return loss

class RnnEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(RnnEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, seq, lengths):
        out = self.encoder(seq)

        out = pack_padded_sequence(out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False)
        _, (hn, _) = self.lstm(out, None)
        
        #out = torch.cat([hn[-1], hn[-2]], dim=1)
        out = hn[-1]
        
        out = self.decoder(out)
        
        return out

class RnnDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, cond_dim, num_layers):
        super(RnnDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(hidden_dim + cond_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, seq, code, lengths):
        code = code.unsqueeze(1).expand(code.size(0), seq.size(1), code.size(1))
        
        out = self.encoder(seq)
        out = torch.cat([out, code], dim=2)

        out = pack_padded_sequence(out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False)
        out, hidden = self.lstm(out, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        logit = self.decoder(out)
        return logit


    def sample(self, code, vocab, mode="sample"):
        code = code.unsqueeze(1)

        batch_size = code.size(0)
        seq = [torch.full((batch_size, 1), vocab.get_start_id(), dtype=torch.long).cuda()]

        hidden = None
        terminated = torch.zeros(batch_size, dtype=torch.bool).cuda()
        log_prob = 0.0
        lengths = torch.ones(batch_size, dtype=torch.long).cuda()
        for _ in range(vocab.max_length):
            out = self.encoder(seq[-1])
            out = torch.cat([out, code], dim=2)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            if mode == "sample":
                seq_t = distribution.sample()
            elif mode == "max":
                seq_t = logit.argmax(dim=2)
                
            log_prob += (~terminated).float() * distribution.log_prob(seq_t).squeeze(1)

            seq.append(seq_t)

            lengths[~terminated] += 1
            terminated = terminated | (seq_t.squeeze(1) == vocab.get_end_id())
            
            if terminated.all():
                break

        seq = torch.cat(seq, dim=1)
        
        return seq, lengths, log_prob
    