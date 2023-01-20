import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import ast

device = "cuda" if torch.cuda.is_available() else "cpu"

class EICUDataSet(Dataset):
    def __init__(self, csv_file_x, csv_file_y):
        self.data_x = pd.read_csv(csv_file_x,index_col='id')
        self.labels = pd.read_csv(csv_file_y, index_col='id')
        self.times = self.data_x.columns.values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.data_x.iloc[idx].apply(ast.literal_eval).values
        x = torch.stack([torch.tensor(y) for y in x])
        y = torch.tensor([self.labels['survival_90days'].loc[idx]]).float()

        return x.to(device), y.to(device)

class MIMICDataSet(Dataset):
    def __init__(self, csv_file_x, csv_file_y):
        self.data_x = pd.read_csv(csv_file_x, index_col='id')
        self.labels = pd.read_csv(csv_file_y, index_col='id')
        self.times = self.data_x.columns.values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.data_x.iloc[idx].apply(ast.literal_eval).values
        x = torch.stack([torch.tensor(y) for y in x])        
        y = torch.tensor([self.labels['90_days_survival'].loc[idx]]).float()

        return x.to(device), y.to(device)