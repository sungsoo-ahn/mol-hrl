import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical


class RnnDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim, code_dim):
        super(RnnDecoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.code_encoder = nn.Linear(code_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, batched_sequence_data, codes):
        sequences, lengths = batched_sequence_data
        codes = codes.unsqueeze(1).expand(-1, sequences.size(1), -1)

        sequences_embedding = self.encoder(sequences)
        codes_embedding = self.code_encoder(codes)

        out = sequences_embedding + codes_embedding

        out = pack_padded_sequence(
            out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False
        )
        out, _ = self.lstm(out, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        logit = self.decoder(out)

        return logit

    def sample(self, codes, start_id, end_id, max_length):
        sample_size = codes.size(0)
        sequences = [torch.full((sample_size, 1), start_id, dtype=torch.long).cuda()]
        hidden = None
        terminated = torch.zeros(sample_size, dtype=torch.bool).cuda()
        log_probs = 0.0
        lengths = torch.ones(sample_size, dtype=torch.long).cuda()

        for _ in range(max_length):
            out = self.encoder(sequences[-1])
            out = out + self.code_encoder(codes).unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)
            
            prob = torch.softmax(logit, dim=2)
            distribution = Categorical(probs=prob)
            tth_sequences = distribution.sample()

            log_probs += (~terminated).float() * distribution.log_prob(tth_sequences).squeeze(1)

            sequences.append(tth_sequences)

            lengths[~terminated] += 1
            terminated = terminated | (tth_sequences.squeeze(1) == end_id)

            if terminated.all():
                break

        sequences = torch.cat(sequences, dim=1)

        return sequences, lengths, log_probs

def rnn_sample_large(model, codes, start_id, end_id, max_length, sample_size, batch_size):
    num_sampling = sample_size // batch_size
    sequences = []
    lengths = []
    log_probs = []
    for _ in range(num_sampling):
        batch_sequences, batch_lengths, batch_log_probs = model.sample(codes, start_id, end_id, max_length)
        sequences.append(batch_sequences)
        lengths.append(batch_lengths)
        log_probs.append(batch_log_probs)

    sequences = torch.cat(sequences, dim=0)
    lengths = torch.cat(lengths, dim=0)
    log_probs = torch.cat(log_probs, dim=0)

    return sequences, lengths, log_probs


