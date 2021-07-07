from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from data.score.dataset import load_scores
from evaluation.util import extract_codes, run_lso

class LinearRegressionModel(torch.nn.Module):
    tag="linear"
    def __init__(self, code_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(code_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
    def neg_score(self, x):
        return -self.linear(x).squeeze(1)

def train_linear(regression_model, train_codes, train_scores, val_codes, val_scores, run):
    regression_model = regression_model.cuda()
    optimizer = torch.optim.Adam(regression_model.parameters(), lr=1e-3)
    
    train_dataset = TensorDataset(train_codes, train_scores)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = TensorDataset(val_codes, val_scores)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    for _ in tqdm(range(100)):
        regression_model.train()
        for codes, scores in train_dataloader:
            codes = codes.cuda()
            scores = scores.cuda()
            
            out = regression_model(codes)
            loss = F.mse_loss(out, scores).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run["lso_linear/train/loss/mse"].log(loss.item())

        regression_model.eval()
        avg_loss = 0.0
        for codes, scores in val_dataloader:
            codes = codes.cuda()
            scores = scores.cuda()
            with torch.no_grad():
                out = regression_model(codes)
            
            loss = F.mse_loss(out, scores).mean()
            avg_loss += loss / len(val_dataloader)
        
        run["lso_linear/val/loss/mse"].log(avg_loss)

def run_lso_linear(model, score_func_name, run):
    model.eval()
    train_codes = extract_codes(model, "train_labeled")
    train_scores = load_scores(model.hparams.data_dir, score_func_name, "train_labeled").view(-1)
    train_scores = (train_scores - train_scores.mean()) / train_scores.std()
    val_codes = extract_codes(model, "val")
    val_scores = load_scores(model.hparams.data_dir, score_func_name, "val").view(-1)
    val_scores = (val_scores - train_scores.mean()) / train_scores.std()
    
    regression_model = LinearRegressionModel(train_codes.size(1))
    train_linear(regression_model, train_codes, train_scores, val_codes, val_scores, run)
    run_lso(model, regression_model, train_codes, train_scores, score_func_name, run)