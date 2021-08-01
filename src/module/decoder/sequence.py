import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
from data.sequence.vocab import START_ID, END_ID, PAD_ID
from data.sequence.util import load_tokenizer, load_vocabulary, sequence2smiles


def compute_sequence_accuracy(logits, batched_sequence_data):
    sequences, _ = batched_sequence_data
    batch_size = sequences.size(0)
    logits = logits[:, :-1]
    targets = sequences[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == PAD_ID] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data):
    sequences, lengths = batched_sequence_data
    logits = logits[:, :-1]
    targets = sequences[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum", ignore_index=PAD_ID,
    )
    loss /= torch.sum(lengths - 1)

    return loss


class SequenceDecoder(nn.Module):
    def __init__(self, hparams):
        super(SequenceDecoder, self).__init__()
        self.hparams = hparams
        self.vocabulary = load_vocabulary(hparams.data_dir)
        self.tokenizer = load_tokenizer(hparams.data_dir)
        num_vocabs = len(self.vocabulary)

        self.encoder = nn.Embedding(num_vocabs, hparams.decoder_hidden_dim)
        self.code_encoder = nn.Linear(hparams.code_dim, hparams.decoder_hidden_dim)
        self.lstm = nn.LSTM(
            hparams.decoder_hidden_dim,
            hparams.decoder_hidden_dim,
            batch_first=True,
            num_layers=hparams.decoder_num_layers,
        )
        self.decoder = nn.Linear(hparams.decoder_hidden_dim, num_vocabs)
        self.max_length = hparams.decoder_max_length

    def forward(self, batched_sequence_data, codes):
        sequences, lengths = batched_sequence_data

        codes = codes.unsqueeze(1).expand(-1, sequences.size(1), -1)

        sequences_embedding = self.encoder(sequences)
        codes_embedding = self.code_encoder(codes)

        out = sequences_embedding + codes_embedding

        out = pack_padded_sequence(out, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False)
        out, _ = self.lstm(out, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.decoder(out)

        return out

    def compute_recon_loss(self, logits, targets):
        loss = compute_sequence_cross_entropy(logits, targets)
        elemwise_acc, acc = compute_sequence_accuracy(logits, targets)
        statistics = {"loss/recon": loss, "acc/elem": elemwise_acc, "acc/seq": acc}

        return loss, statistics

    def sample(self, codes, argmax):
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
            if argmax == True:
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

    def sample_smiles(self, codes, argmax):
        sequences, lengths = self.sample(codes, argmax)
        sequences = sequences.cpu()
        lengths = lengths.cpu()
        sequences = [sequence[:length] for sequence, length in zip(sequences, lengths)]
        smiles_list = [sequence2smiles(sequence, self.tokenizer, self.vocabulary) for sequence in sequences]
        return smiles_list
