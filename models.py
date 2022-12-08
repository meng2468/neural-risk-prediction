import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class BaseRecurrent(nn.Module):
    def __init__(self):
        super(BaseRecurrent, self).__init__()
        self.hidden_size = 60
        self.flatten = nn.Flatten()
        self.recurrent = nn.RNN(input_size=33, hidden_size=self.hidden_size)
        self.relu = nn.ReLU()
        self.final = nn.Linear(in_features=self.hidden_size,out_features=2)

    def forward(self, x):
        out, hidden = self.recurrent(x)
        logits = (self.final(out[:,-1,:]))
        return logits