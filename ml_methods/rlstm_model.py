#! /usr/bin/env python3


import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


# The main model for predicting straggling time
class ShallowRegressionLSTM(nn.Module):
    # input size is 3: available CPU difference, available bw differece, straggling time
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size  # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=self.num_layers)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


# Other functional code for the model
class SequenceDataset(Dataset):
    def __init__(self, seq, win_len=10):
        self.seq = copy.deepcopy(seq)
        self.win_len = win_len
        self.seq_x = torch.tensor(np.array(self.seq[:-1], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)
        self.seq_y = torch.tensor(np.array(self.seq[1:], dtype=np.float32)[:, np.newaxis], dtype=torch.float32)

    def __len__(self):
        return len(self.seq) - 1

    def __getitem__(self, i):
        if i >= self.win_len - 1:
            i_start = i - self.win_len + 1
            x = self.seq_x[i_start : (i + 1), :]
        else:
            padding = self.seq_x[0].repeat(self.win_len - i - 1, 1)
            x = self.seq_x[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.seq_y[i][0]


def train_model(data_loaders, model, loss_function, optimizer):
    num_batches = 0
    for loader in data_loaders:
        num_batches += len(loader)
    total_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    model = model.to(device)
    model.train()

    loss_function = loss_function.to(device)

    for data_loader in data_loaders:
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

    return model


def predict(model, seq):
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    model = model.to(device)
    model.eval()

    input = np.array(seq, dtype=np.float32)[:, np.newaxis]
    input = np.reshape(input, (1, len(input), 1))
    input = torch.tensor(input, dtype=torch.float32).to(device)

    # print(f"model is on cuda {model.is_cuda}")
    # print(f"tensor is on cuda {input.is_cuda}")

    with torch.no_grad():
        output = model(input)

    return float(output.to("cpu")[0])
