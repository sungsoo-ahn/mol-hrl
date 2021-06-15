import torch
import torch.nn.functional as F

class HigherPolicy(torch.nn.Module):
    def __init__(self, code_dim):
        super(HigherPolicy, self).__init__()
        self.mu = torch.nn.Parameter(torch.zeros(code_dim))
        self.var = torch.nn.Parameter(torch.zeros(code_dim))
    
    def sample(self, sample_size):
        std = torch.exp(self.var / 2)
        distribution = torch.distributions.Normal(self.mu, std)
        code = distribution.rsample([sample_size])
        log_prob = distribution.log_prob(code)
        return code, log_prob

    def log_prob(self, code):
        std = torch.exp(self.var / 2)
        distribution = torch.distributions.Normal(self.mu, std)
        log_prob = distribution.log_prob(code)
        return log_prob

class HierarchicalPolicy(torch.nn.Module):
    def __init__(self, encoder, decoder, code_dim):
        super(HierarchicalPolicy, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.higher_policy = HigherPolicy(code_dim)
        self.code_dim = code_dim

    def decoder_step(self, sample_size, vocab):
        code, _ = self.higher_policy.sample(sample_size)

        seqs, lengths, log_probs = self.decoder.sample(code, vocab, mode="sample")
        recon_code = self.encoder(seqs, lengths)

        decoder_reward = -F.mse_loss(code, recon_code).mean()

        loss = -torch.mean(decoder_reward * log_probs)

        statistics = {
            "loss/hagent/decoder": loss.item(),
            "reward/hagent/decoder": decoder_reward.mean().item()
        }

        return loss, statistics

        