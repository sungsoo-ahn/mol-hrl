import torch

from model.rnn import compute_rnn_accuracy, compute_rnn_ce

class RnnVariationalAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, code_dim):
        super(RnnVariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = torch.nn.Linear(code_dim, code_dim)
        self.fc_var = torch.nn.Linear(code_dim, code_dim)
        self.code_dim = code_dim
        self.kl_coef = 1e-2

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def sample_seq(self, sample_size, vocab):
        distribution = torch.distributions.Normal(0.0, 1.0)
        codes = distribution.sample([sample_size, self.code_dim]).cuda()
        seqs, lengths, log_probs = self.decoder.sample(codes, vocab, mode="sample")
        return seqs, lengths, log_probs

    def run_step(self, x_seq, lengths):
        x = self.encoder(x_seq, lengths)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        logits = self.decoder(x_seq[:, :-1], code=z, lengths=lengths-1)
        
        return z, logits, p, q

    def step(self, x_seq, lengths):
        z, logits, p, q = self.run_step(x_seq, lengths)
        y_seq = x_seq[:, 1:]

        recon_loss = compute_rnn_ce(logits, y_seq, lengths)
        recon_elem_acc, recon_acc = compute_rnn_accuracy(logits, y_seq, z.size(0))
        
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl_loss = log_qz - log_pz
        kl_loss = kl_loss.mean()
        
        loss = self.kl_coef * kl_loss + recon_loss 

        statistics = {
            "loss/total": loss.item(),
            "loss/recon": recon_loss.item(),
            "loss/kl": kl_loss.item(),
            "acc/recon/elem": recon_elem_acc.item(),
            "acc/recon/seq": recon_acc.item(),
        }

        return loss, statistics

class EducatedRnnVariationalAutoEncoder(RnnVariationalAutoEncoder):
    def run_step(self, seqs0, seqs1, lengths0, lengths1):
        x = self.encoder(seqs0, lengths0)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, code = self.sample(mu, log_var)

        logits = self.decoder(seqs1[:, :-1], code=code, lengths=lengths1-1)
        
        return code, logits, p, q

    def step(self, seqs0, seqs1, lengths0, lengths1):
        z, logits, p, q = self.run_step(seqs0, seqs1, lengths0, lengths1)
        y_seq = seqs1[:, 1:]

        recon_loss = compute_rnn_ce(logits, y_seq, lengths1)
        recon_elem_acc, recon_acc = compute_rnn_accuracy(logits, y_seq, z.size(0))
        
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl_loss = log_qz - log_pz
        kl_loss = kl_loss.mean()
        
        loss = self.kl_coef * kl_loss + recon_loss 

        statistics = {
            "loss/total": loss.item(),
            "loss/recon": recon_loss.item(),
            "loss/kl": kl_loss.item(),
            "acc/recon/elem": recon_elem_acc.item(),
            "acc/recon/seq": recon_acc.item(),
        }

        return loss, statistics



