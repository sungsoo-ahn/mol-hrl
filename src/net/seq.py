import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
from data.seq.util import START_ID, END_ID, PAD_ID


def compute_sequence_accuracy(logits, batched_sequence_data):
    sequences, _ = batched_sequence_data
    batch_size = sequences.size(0)
    logits = logits[:, :-1]
    targets = sequences[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == PAD_ID] = True
    elem_acc = correct[targets != 0].float().mean()
    seq_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, seq_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data):
    sequences, lengths = batched_sequence_data
    logits = logits[:, :-1]
    targets = sequences[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="sum",
        ignore_index=PAD_ID,
    )
    loss /= torch.sum(lengths - 1)

    return loss


class SeqEncoder(nn.Module):
    def __init__(self, hparams):
        super(SeqEncoder, self).__init__()
        self.encoder = nn.Embedding(hparams.num_vocabs, hparams.encoder_hidden_dim)
        self.lstm = nn.LSTM(
            hparams.encoder_hidden_dim, 
            hparams.encoder_hidden_dim, 
            batch_first=True, 
            num_layers=hparams.encoder_num_layers,
            )
        self.decoder = nn.Linear(2 * hparams.encoder_hidden_dim, hparams.code_dim)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--encoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--encoder_num_layers", type=int, default=3)
        parser.add_argument("--code_dim", type=int, default=256)
        
    def forward(self, batched_sequence_data):
        sequences, lengths = batched_sequence_data
        out = self.encoder(sequences)
        out = pack_padded_sequence(
            out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False
        )
        _, (h, _) = self.lstm(out, None)
        out = torch.cat([h[-2], h[-1]], 1)
        out = self.decoder(out)
        
        return out, {}

class SeqDecoder(nn.Module):
    def __init__(self, hparams):
        super(SeqDecoder, self).__init__()
        self.encoder = nn.Embedding(hparams.num_vocabs, hparams.decoder_hidden_dim)
        self.code_encoder = nn.Linear(hparams.code_dim, hparams.decoder_hidden_dim)
        self.lstm = nn.LSTM(
            hparams.decoder_hidden_dim, 
            hparams.decoder_hidden_dim, 
            batch_first=True, 
            num_layers=hparams.decoder_num_layers,
            )
        self.decoder = nn.Linear(hparams.decoder_hidden_dim, hparams.num_vocabs)
        self.max_length = hparams.decoder_max_length

    @staticmethod
    def add_args(parser):
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--decoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_max_length", type=int, default=81)
        
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

        logits = self.decoder(out)

        loss = compute_sequence_cross_entropy(logits, batched_sequence_data)
        elemwise_acc,acc = compute_sequence_accuracy(logits, batched_sequence_data)
        statistics = {"elemwise_acc": elemwise_acc, "acc": acc}
        return loss, statistics

    def decode(self, codes, deterministic):
        sample_size = codes.size(0)
        sequences = [torch.full((sample_size, 1), START_ID, dtype=torch.long).cuda()]
        hidden = None
        terminated = torch.zeros(sample_size, dtype=torch.bool).cuda()
        lengths = torch.ones(sample_size, dtype=torch.long).cuda()

        for _ in range(self.max_length):
            out = self.encoder(sequences[-1])
            out = out + self.code_encoder(codes).unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)

            prob = torch.softmax(logit, dim=2)
            if deterministic==True:
                tth_sequences = torch.argmax(logit, dim=2)
            else:
                distribution = Categorical(probs=prob)
                tth_sequences = distribution.sample()

            sequences.append(tth_sequences)

            lengths[~terminated] += 1
            terminated = terminated | (tth_sequences.squeeze(1) == END_ID)

            if terminated.all():
                break

        sequences = torch.cat(sequences, dim=1)

        return sequences, lengths

"""
def rnn_sample_large(model, codes, start_id, end_id, max_length, sample_size, batch_size):
    num_sampling = sample_size // batch_size
    sequences = []
    lengths = []
    log_probs = []
    for _ in range(num_sampling):
        batch_sequences, batch_lengths, batch_log_probs = model.sample(
            codes, start_id, end_id, max_length
        )
        sequences.append(batch_sequences)
        lengths.append(batch_lengths)
        log_probs.append(batch_log_probs)

    sequences = torch.cat(sequences, dim=0)
    lengths = torch.cat(lengths, dim=0)
    log_probs = torch.cat(log_probs, dim=0)

    return sequences, lengths, log_probs
"""