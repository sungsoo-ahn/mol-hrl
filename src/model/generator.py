import torch

from model.rnn import Rnn, GoalEncoderRnn, GoalDecoderRnn


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

    loss = torch.nn.functional.cross_entropy(logits, y_seq, reduction="sum", ignore_index=pad_id)
    loss /= torch.sum(lengths - 1)

    return loss


class BaseGenerator(torch.nn.Module):
    dataset_type = "base"

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
    

class GoalBasedGenerator(torch.nn.Module):
    dataset_type = "base"

    def __init__(self, hidden_dim, goal_dim, num_layers, vocab):
        super(GoalBasedGenerator, self).__init__()
        self.encoder = GoalEncoderRnn(len(vocab), goal_dim, hidden_dim, num_layers)
        self.decoder = GoalDecoderRnn(len(vocab), len(vocab), hidden_dim, goal_dim, num_layers)
        self.global_goal = torch.nn.Parameter(torch.randn(goal_dim))
        self.start_id = vocab.get_start_id()
        self.end_id = vocab.get_end_id()
        self.pad_id = vocab.get_pad_id()
        self.max_length = vocab.get_max_length()

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("generator")
        group.add_argument("--generator_hidden_dim", type=int, default=1024)
        group.add_argument("--generator_goal_dim", type=int, default=32)
        group.add_argument("--generator_num_layers", type=int, default=3)
        return parser

    def pretrain_step(self, batched_data):
        seqs, lengths = batched_data
        seqs = seqs.cuda()
        lengths = lengths.cuda()

        batch_size = seqs.size(0)
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        goal = self.encoder(seqs, lengths)
        global_logits, logits = self.decoder(
            seqs[:, :-1], goals=[global_goal, goal], lengths=lengths - 1
        )

        global_recon_loss = compute_cross_entropy(global_logits, seqs[:, 1:], lengths, self.pad_id)
        recon_loss = compute_cross_entropy(logits, seqs[:, 1:], lengths, self.pad_id)
        elem_acc, seq_acc = compute_accuracy(logits, seqs[:, 1:], batch_size, self.pad_id)

        loss = global_recon_loss + recon_loss

        statistics = {
            "loss/sum": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
            "loss/recon": recon_loss.item(),
            "acc/recon/elem": elem_acc.item(),
            "acc/recon/seq": seq_acc.item(),
        }

        return loss, statistics

    def hillclimb_step(self, batched_data):
        seqs, lengths = batched_data
        seqs = seqs.cuda()
        lengths = lengths.cuda()

        batch_size = seqs.size(0)
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        (global_logits,) = self.decoder(seqs[:, :-1], goals=[global_goal], lengths=lengths - 1)

        global_recon_loss = compute_cross_entropy(global_logits, seqs[:, 1:], lengths, self.pad_id)

        loss = global_recon_loss

        statistics = {
            "loss/sum": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
        }

        return loss, statistics

    def goal_based_sample(self, sample_size, goal):
        return self.decoder.sample(sample_size, goal, self.start_id, self.end_id, self.max_length)

    def sample(self, sample_size):
        global_goal = self.global_goal.unsqueeze(0).expand(sample_size, -1)
        seqs, lengths, log_prob = self.decoder.sample(
            sample_size, global_goal, self.start_id, self.end_id, self.max_length
        )
        return seqs, lengths, log_prob

    def get_pretrain_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_hillclimb_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
class EducatedGoalBasedGenerator(GoalBasedGenerator):
    dataset_type = "paired"

    def pretrain_step(self, batched_data):
        (seqs0, lengths0), (seqs1, lengths1) = batched_data
        seqs0 = seqs0.cuda()
        lengths0 = lengths0.cuda()
        seqs1 = seqs1.cuda()
        lengths1 = lengths1.cuda()

        batch_size = seqs0.size(0)
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        goal = self.encoder(seqs0, lengths0)

        global_logits, logits = self.decoder(
            seqs1[:, :-1], goals=[global_goal, goal], lengths=lengths1 - 1
        )

        global_recon_loss = compute_cross_entropy(
            global_logits, seqs1[:, 1:], lengths1, self.pad_id
        )
        recon_loss = compute_cross_entropy(logits, seqs1[:, 1:], lengths1, self.pad_id)
        elem_acc, seq_acc = compute_accuracy(logits, seqs1[:, 1:], batch_size, self.pad_id)

        loss = global_recon_loss + recon_loss

        statistics = {
            "loss/sum": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
            "loss/recon": recon_loss.item(),
            "acc/recon/elem": elem_acc.item(),
            "acc/recon/seq": seq_acc.item(),
        }

        return loss, statistics
