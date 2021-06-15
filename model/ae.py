import torch

from model.rnn import compute_rnn_accuracy, compute_rnn_ce

class RnnAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, code_dim):
        super(RnnAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = torch.nn.Linear(code_dim, code_dim)
        self.fc_var = torch.nn.Linear(code_dim, code_dim)
        self.code_dim = code_dim
        self.kl_coef = 1e-1

    def step(self, x_seq, lengths):
        code = self.encoder(x_seq, lengths)
        logits = self.decoder(x_seq[:, :-1], code=code, lengths=lengths-1)
        recon_loss = compute_rnn_ce(logits, x_seq[:, 1:], lengths)
        elem_acc, seq_acc = compute_rnn_accuracy(logits, x_seq[:, 1:], code.size(0))
        
        loss = recon_loss

        statistics = {
            "loss/total": loss.item(),
            "loss/recon": recon_loss.item(),
            "acc/recon/elem": elem_acc.item(),
            "acc/recon/seq": seq_acc.item(),
        }

        return loss, statistics    