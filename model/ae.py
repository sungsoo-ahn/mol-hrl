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

class EducatedRnnAutoEncoder(RnnAutoEncoder):
    def step(self, x_seq0, x_seq1, lengths0, lengths1):
        batch_size = x_seq0.size(0)        
        global_goal = self.global_goal.unsqueeze(0).expand(batch_size, -1)
        goal0 = self.encoder(x_seq0, lengths0)
        goal1 = self.encoder(x_seq1, lengths1)
        
        global_logits0, logits0 = self.decoder(x_seq0[:, :-1], goal_list=[global_goal, goal1], lengths=lengths0-1)
        global_logits1, logits1 = self.decoder(x_seq1[:, :-1], goal_list=[global_goal, goal0], lengths=lengths1-1)
        
        global_recon_loss = 0.5 * (
            compute_rnn_ce(global_logits0, x_seq0[:, 1:], lengths0) 
            + compute_rnn_ce(global_logits1, x_seq1[:, 1:], lengths1)
        )
        recon_loss = 0.5 * (
            compute_rnn_ce(logits0, x_seq0[:, 1:], lengths0)
            + compute_rnn_ce(logits1, x_seq1[:, 1:], lengths1)
        )
        elem_acc0, seq_acc0 = compute_rnn_accuracy(logits0, x_seq0[:, 1:], batch_size)
        elem_acc1, seq_acc1 = compute_rnn_accuracy(logits1, x_seq1[:, 1:], batch_size)
        elem_acc = 0.5 * (elem_acc0 + elem_acc1)
        seq_acc = 0.5 * (seq_acc0 + seq_acc1)
        loss = global_recon_loss + recon_loss

        statistics = {
            "loss/total": loss.item(),
            "loss/global_recon": global_recon_loss.item(),
            "loss/recon": recon_loss.item(),
            "acc/recon/elem": elem_acc.item(),
            "acc/recon/seq": seq_acc.item(),
        }

        return loss, statistics