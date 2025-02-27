import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from prep import load_seq

class Load_X(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

class Load_Xy(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

class RMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.weights = weights

    def forward(self, input, target):
        mask = torch.isnan(target)
        no_nan_counts = (~mask).sum(dim=0, keepdim=True).clamp_min(1)
        input = input.masked_fill(mask, 0)
        target = target.masked_fill(mask, 0)

        mse_loss = self.mse_loss(input, target).sum(dim=0, keepdim=True) / no_nan_counts
        rmse_loss = torch.sqrt(mse_loss.clamp_min(1e-8)) * self.weights

        return rmse_loss.sum()
    

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for X, y in dataloader:
        X, y = X.to(config.device), y.to(config.device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= len(dataloader)
    return epoch_loss

def validate(dataloader, model, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.device), y.to(config.device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y).detach().item()
            
    loss /= len(dataloader)      
    return loss


def predict(model, data):
    X, _ = load_seq(data, None)
    dataloader = DataLoader(Load_X(X), batch_size=config.batch_size, shuffle=False)
    
    model.eval()
    pred = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(config.device)
            pred.append(model(x).detach().cpu().numpy())
    pred = np.vstack(pred)

    return pred 