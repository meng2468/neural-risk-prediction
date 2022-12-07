import torch
from torch import nn

from dataloader import EICUDataSet
from torch.utils.data import Dataset, DataLoader
from models import BaseRecurrent

from optimization import train_model, test_loop

if __name__ == '__main__':
    learning_rate = 1e-2
    batch_size = 64
    epochs = 7

    loss_fn =  nn.CrossEntropyLoss()

    data = EICUDataSet('eicu_prepared_x.csv','eicu_prepared_y.csv')
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model = BaseRecurrent()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_model(loader, model, loss_fn, optimizer)
        test_loop(loader, model, loss_fn)

