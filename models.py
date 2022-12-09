import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class BaseRecurrent(nn.Module):
    def __init__(self):
        super(BaseRecurrent, self).__init__()
        self.hidden_size = 120
        self.recurrent = nn.RNN(input_size=33, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=2)

    def forward(self, x):
        out, hidden = self.recurrent(x)
        logits = (self.final(out[:,-1,:]))
        return logits

class LayeredRecurrent(nn.Module):
    def __init__(self):
        super(LayeredRecurrent, self).__init__()
        self.hidden_size = 120
        self.recurrent = nn.RNN(input_size=33, hidden_size=self.hidden_size, num_layers=3, dropout=.5)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=2)

    def forward(self, x):
        out, hidden = self.recurrent(x)
        logits = (self.final(out[:,-1,:]))
        return logits