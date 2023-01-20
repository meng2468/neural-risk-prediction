import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

class DiPole(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(DiPole, self).__init__()
        self.hidden_size = h_size
        self.recurrent = nn.RNN(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.recurrent(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits

class BaseRecurrent(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseRecurrent, self).__init__()
        self.hidden_size = h_size
        self.recurrent = nn.RNN(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.recurrent(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits


class BaseLSTM(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseLSTM, self).__init__()
        self.hidden_size = h_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits

class BaseGRU(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseGRU, self).__init__()
        self.hidden_size = h_size
        self.lstm = nn.GRU(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits


