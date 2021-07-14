from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.seq.dataset import SequenceDataset
from data.graph.dataset import GraphDataset
from data.score.dataset import ScoreDataset
from data.score.factory import get_scoring_func

from lso.nn import train_nn

import torch.nn as nn

class CondVAE(nn.Module):
    def __init__(self, code_dim, hidden_dim):
        self.predictor = nn.Linear(code_dim, 1) 
        self.encoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim)
        )
        self.mu_linear = nn.Linear(code_dim, code_dim)
        self.logstd_linear = nn.Linear(code_dim, code_dim)


    def compute_loss(self, batched_data):
        loss, statistics = 0.0, dict()
        z, loss, statistics = self.update_encoder_loss(batched_data, loss, statistics)
        loss, statistics = self.update_decoder_loss(batched_data, z, loss, statistics)
        return loss, statistics
    
    def update_encoder_loss(self, batched_data, loss, statistics):
        batched_input_data, y = batched_data
        encoder_out = self.encoder(batched_input_data)


        y_pred = self.predictor(batched_input_data)

        return codes, loss, statistics
    
    def update_decoder_loss(self, batched_data, z, loss, statistics):
        batched_input_data, y = batched_data
        decoder_out = self.decoder(z)
        recon_loss, recon_statistics = F.mse_loss(decoder_out, batched_input_data)
        
        loss += recon_loss
        statistics.update(recon_statistics)

        return loss, statistics
        
    