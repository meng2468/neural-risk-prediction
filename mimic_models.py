import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

class BaseRecurrent(nn.Module):
    def __init__(self):
        super(BaseRecurrent, self).__init__()
        self.hidden_size = 300
        self.recurrent = nn.RNN(input_size=51, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.recurrent(x)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits

class LayeredRecurrent(nn.Module):
    def __init__(self):
        super(LayeredRecurrent, self).__init__()
        self.hidden_size = 300
        self.recurrent = nn.RNN(input_size=51, hidden_size=self.hidden_size, num_layers=3, dropout=.2)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.recurrent(x)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits


class BaseLSTM(nn.Module):
    def __init__(self):
        super(BaseLSTM, self).__init__()
        self.hidden_size=300
        self.lstm = nn.LSTM(input_size=51, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits

class BaseGRU(nn.Module):
    def __init__(self):
        super(BaseGRU, self).__init__()
        self.hidden_size=300
        self.lstm = nn.GRU(input_size=51, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits


