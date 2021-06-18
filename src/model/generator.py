import torch

from model.rnn import Rnn, ConditionalRnn
from scoring.featurizer import PENALIZED_LOGP_FEATURE_DIM, compute_penalized_logp_feature
from vocabulary import seq2smiles
from util.mol import is_valid_smiles

def compute_accuracy(logits, y, batch_size, pad_id):
    y_pred = torch.argmax(logits, dim=-1)
    correct = y_pred == y
    correct[y == pad_id] = True
    elem_acc = correct[y != 0].float().mean()
    seq_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, seq_acc


def compute_cross_entropy(logits, y_seq, lengths, pad_id):
    logits = logits.view(-1, logits.size(-1))
    y_seq = y_seq.reshape(-1)

    loss = torch.nn.functional.cross_entropy(
        logits, y_seq, reduction="sum", ignore_index=pad_id
    )
    loss /= torch.sum(lengths - 1)

    return loss


class BaseGenerator(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, vocab):
        super(BaseGenerator, self).__init__()
        self.rnn = Rnn(len(vocab), len(vocab), hidden_dim, num_layers)
        self.start_id = vocab.get_start_id()
        self.end_id = vocab.get_end_id()
        self.pad_id = vocab.get_pad_id()
        self.max_length = vocab.get_max_length()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("generator")
        group.add_argument("--generator_hidden_dim", type=int, default=1024)
        group.add_argument("--generator_num_layers", type=int, default=3)
        group.add_argument("--generator_pretrain_lr", type=float, default=1e-3)
        return parser

    def pretrain_step(self, batched_data):
        seqs, lengths = batched_data
        seqs = seqs.cuda()
        lengths = lengths.cuda()

        logits = self.rnn(seqs[:, :-1], lengths=lengths - 1)

        loss = compute_cross_entropy(logits, seqs[:, 1:], lengths, self.pad_id)

        statistics = {"loss/sum": loss.item()}

        return loss, statistics

    def hillclimb_step(self, batched_data):
        return self.pretrain_step(batched_data)

    def sample(self, sample_size):
        return self.rnn.sample(sample_size, self.start_id, self.end_id, self.max_length)

    def get_pretrain_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_hillclimb_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class PenalizedLogPFeatureBasedGenerator(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, vocab, tokenizer):
        super(PenalizedLogPFeatureBasedGenerator, self).__init__()
        self.decoder = ConditionalRnn(
            len(vocab), len(vocab), hidden_dim, PENALIZED_LOGP_FEATURE_DIM, num_layers
        )
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.start_id = vocab.get_start_id()
        self.end_id = vocab.get_end_id()
        self.pad_id = vocab.get_pad_id()
        self.max_length = vocab.get_max_length()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("generator")
        group.add_argument("--generator_hidden_dim", type=int, default=1024)
        group.add_argument("--generator_num_layers", type=int, default=3)
        return parser

    def step(self, batched_data):
        (seqs, lengths), features = batched_data
        seqs = seqs.cuda()
        lengths = lengths.cuda()
        features = features.cuda()

        logits = self.decoder(seqs[:, :-1], features, lengths - 1)
        loss = compute_cross_entropy(logits, seqs[:, 1:], lengths, self.pad_id)
        
        statistics = {"loss": loss.item()}

        return loss, statistics

    def eval(self, batched_data):
        _, features = batched_data
        features = features.cuda()

        seqs, _, _ = self.decoder.sample(
            features.size(0), features.unsqueeze(1), self.start_id, self.end_id, self.max_length
        )

        smiles_list = seq2smiles(seqs, self.tokenizer, self.vocab)
        valid_idxs = [idx for idx, smiles in enumerate(smiles_list) if is_valid_smiles(smiles)]
        valid_ratio = float(len(valid_idxs)) / len(smiles_list)

        statistics = {"valid_ratio": valid_ratio}
        if len(valid_idxs) > 0: 
            valid_smiles_list = [smiles_list[idx] for idx in valid_idxs]
            valid_features = features[valid_idxs]

            for smiles in valid_smiles_list:
                try:
                    compute_penalized_logp_feature(smiles)
                except:
                    print(smiles)
                    print(is_valid_smiles(smiles))
                    import rdkit
                    print(rdkit.Chem.MolFromSmiles(smiles))
                    assert False
                
            recon_valid_features = torch.stack(
                [torch.tensor(compute_penalized_logp_feature(smiles)) for smiles in valid_smiles_list], dim=0
            ).cuda()

            error = torch.nn.functional.mse_loss(valid_features, recon_valid_features)
            statistics["error"] = error.item()

        return statistics

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)