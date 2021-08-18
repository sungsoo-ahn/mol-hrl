import torch
import torch.nn as nn
from torch.distributions import Categorical
from data.util import load_tokenizer

class LSTMDecoder(nn.Module):
    def __init__(self, decoder_num_layers, decoder_hidden_dim, code_dim):
        super(LSTMDecoder, self).__init__()
        self.tokenizer = load_tokenizer()
        num_vocabs = self.tokenizer.get_vocab_size()

        self.encoder = nn.Embedding(num_vocabs, decoder_hidden_dim)
        self.code_encoder = nn.Linear(code_dim, decoder_hidden_dim)
        self.lstm = nn.LSTM(
            decoder_hidden_dim,
            decoder_hidden_dim,
            batch_first=True,
            num_layers=decoder_num_layers,
        )
        self.decoder = nn.Linear(decoder_hidden_dim, num_vocabs)
        
    def forward(self, batched_sequence_data, codes):
        codes = codes.unsqueeze(1).expand(-1, batched_sequence_data.size(1), -1)
        sequences_embedding = self.encoder(batched_sequence_data)
        codes_embedding = self.code_encoder(codes)

        out = sequences_embedding + codes_embedding
        out, _ = self.lstm(out, None)
        out = self.decoder(out)

        return out

    def sample(self, codes, argmax, max_len):
        sample_size = codes.size(0)
        sequences = [torch.full((sample_size, 1), self.tokenizer.token_to_id("[BOS]"), dtype=torch.long).cuda()]
        hidden = None
        terminated = torch.zeros(sample_size, dtype=torch.bool).cuda()
        lengths = torch.ones(sample_size, dtype=torch.long).cuda()

        code_encoder_out = self.code_encoder(codes)
        for _ in range(max_len):
            out = self.encoder(sequences[-1])
            out = out + code_encoder_out.unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            if argmax == True:
                tth_sequences = torch.argmax(logit, dim=2)
            else:
                distribution = Categorical(probs=prob)
                tth_sequences = distribution.sample()

            tth_sequences[terminated] = self.tokenizer.token_to_id("[PAD]")
            sequences.append(tth_sequences)

            lengths[~terminated] += 1
            terminated = terminated | (tth_sequences.squeeze(1) == self.tokenizer.token_to_id("[EOS]"))

            if terminated.all():
                break

        sequences = torch.cat(sequences, dim=1)

        return sequences