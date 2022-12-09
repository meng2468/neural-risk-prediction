import torch
from torch import nn

from dataloader import EICUDataSet
from torch.utils.data import Dataset, DataLoader
from models import BaseRecurrent

from optimization import train_model, test_loop

if __name__ == '__main__':
    learning_rate = 1e-2
    batch_size = 64
    epochs = 6

    loss_fn =  nn.CrossEntropyLoss()

    train_data = EICUDataSet('data/eicu_train_x.csv','data/eicu_train_y.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = EICUDataSet('data/eicu_test_x.csv','data/eicu_test_y.csv')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    val_data = EICUDataSet('data/eicu_val_x.csv','data/eicu_val_y.csv')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = BaseRecurrent()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print('*'*20)
        print('Running Epoch', epoch)
        
        train_model(train_loader, model, loss_fn, optimizer)
        test_loop(val_loader, model, loss_fn)
        test_loop(test_loader, model, loss_fn)

    torch.save(model.state_dict, 'base_model.ts')
