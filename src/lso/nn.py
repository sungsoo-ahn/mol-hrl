from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

class LinearRegressionModel(nn.Module):
    def __init__(self, code_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(code_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
    def neg_score(self, x):
        return -self.linear(x).squeeze(1)

def train_nn(
    train_codes, 
    val_codes, 
    train_score_dataset, 
    val_score_dataset, 
    score_func_name, 
    run, 
    batch_size=256,
    lr=1e-3,
    epochs=100,
    ):
    model = LinearRegressionModel(train_codes.size(1))
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(train_codes, train_score_dataset.tsrs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_codes, val_score_dataset.tsrs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def shared_step(model, codes, scores):
        codes = codes.cuda()
        scores = scores.cuda()
        
        out = model(codes)
        loss = F.mse_loss(out, scores.squeeze(1)).mean()
        return loss

    for _ in tqdm(range(epochs)):
        # Train
        model.train()
        for codes, scores in train_dataloader:
            loss = shared_step(model, codes, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run[f"lso_linear/{score_func_name}/train/loss/mse"].log(loss.item())

        # Evaluation
        model.eval()
        avg_loss = 0.0
        for codes, scores in val_dataloader:
            with torch.no_grad():
                loss = shared_step(model, codes, scores)
            
            avg_loss += loss / len(val_dataloader)
        
        run[f"lso_linear/{score_func_name}/val/loss/mse"].log(avg_loss)

    return model