import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class EICUDataSet(Dataset):
    def __init__(self, csv_file_x, csv_file_y):
        self.data_x = pd.read_csv(csv_file_x,index_col='id')
        self.labels = pd.read_csv(csv_file_y, index_col='id')
        self.times = self.data_x.t.unique()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = np.array([self.data_x.loc[idx].query('t == '+str(x)).value.values for x in self.times], dtype='f')
        if self.labels.loc[idx]['survival_90days'] == 'alive':
            y = 1
        else:
            y = 0
        return x, y

