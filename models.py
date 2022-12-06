import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class BaseRecurrent(nn.Module):
    def __init__(self):
        super(BaseRecurrent, self).__init__()
        self.flatten = nn.Flatten()
        self.recurrent = nn.RNN(input_size=33, hidden_size=30)
        self.relu = nn.ReLU()
        self.final = nn.Linear(in_features=30,out_features=2)

    def forward(self, x):
        hidden_state = self.recurrent(x)
        logits = self.relu(self.final(hidden_state[-1]))
        return logits