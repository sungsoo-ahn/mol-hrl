import torch

from model.rnn import compute_rnn_accuracy, compute_rnn_ce

class RnnAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, global_goal):
        super(RnnAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.global_goal = global_goal
        
    def step(self, x_seq, lengths):
        batch_size = x_seq.size(0)        
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        goal = self.encoder(x_seq, lengths)
        global_logits, logits = self.decoder(x_seq[:, :-1], goal_list=[global_goal, goal], lengths=lengths-1)
        
        global_recon_loss = compute_rnn_ce(global_logits, x_seq[:, 1:], lengths)
        recon_loss = compute_rnn_ce(logits, x_seq[:, 1:], lengths)
        elem_acc, seq_acc = compute_rnn_accuracy(logits, x_seq[:, 1:], batch_size)
        
        loss = global_recon_loss + recon_loss

        statistics = {
            "loss/total": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
            "loss/recon": recon_loss.item(),
            "acc/recon/elem": elem_acc.item(),
            "acc/recon/seq": seq_acc.item(),
        }

        return loss, statistics    
    
    def global_step(self, x_seq, lengths):
        batch_size = x_seq.size(0)        
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        global_logits, = self.decoder(x_seq[:, :-1], goal_list=[global_goal], lengths=lengths-1)
        
        global_recon_loss = compute_rnn_ce(global_logits, x_seq[:, 1:], lengths)
        
        loss = global_recon_loss

        statistics = {
            "loss/total": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
        }

        return loss, statistics
    
    def global_sample(self, sample_size, vocab):
        global_goal = self.global_goal.unsqueeze(0).expand(sample_size, -1)
        x_seq, lengths, log_prob = self.decoder.sample(global_goal, vocab)
        return x_seq, lengths, log_prob

        