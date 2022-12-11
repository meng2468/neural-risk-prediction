import torch
from torch import nn
import sys

from dataloader import EICUDataSet
from torch.utils.data import DataLoader
from models import BaseRecurrent, LayeredRecurrent
from models import BaseLSTM, BaseGRU

from optimization import train_model, test_loop
from evaluation import save_plot_loss

def run_train_test(model, model_name, learning_rate, batch_size, epochs):
    print('Running training for '+model_name)
    # loss_fn =  nn.CrossEntropyLoss(weight=torch.tensor([.8,.2]))
    loss_fn =  nn.BCELoss()

    train_data = EICUDataSet('data/eicu_train_x.csv','data/eicu_train_y.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = EICUDataSet('data/eicu_test_x.csv','data/eicu_test_y.csv')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    val_data = EICUDataSet('data/eicu_val_x.csv','data/eicu_val_y.csv')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = [[],[]]
    accurracies = [[],[]]
    for epoch in range(epochs):
        print('*'*20)
        print('Running Epoch', epoch)
        
        ts_l, ts_a = test_loop(val_loader, model, loss_fn)
        tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)

        losses[0].append(tr_l)
        losses[1].append(ts_l)

        accurracies[0].append(tr_a)
        accurracies[1].append(ts_a)
        
        save_plot_loss(losses[0], losses[1], model_name)

        # Early stopping
        if max(losses[1][-3:]) == losses[1][-1] and len(losses[1]) > 3:
            print('Stopping model training early', losses)
            break
        

    torch.save(model.state_dict, 'models/'+model_name+'.model')
    

if __name__ == '__main__':
    learning_rate = 1e-5
    batch_size = 50
    epochs = 40
    
    model = BaseGRU()
    model_name = 'wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/base_gru'
    run_train_test(model, model_name, learning_rate, batch_size, 15)

    model = LayeredRecurrent()
    model_name = 'layered_rnn'
    run_train_test(model, model_name, learning_rate, batch_size, 15)
    
    model = BaseLSTM()
    model_name = 'base_lstm'
    run_train_test(model, model_name, learning_rate, batch_size, 15)

    model = BaseRecurrent()
    model_name = 'base_rnn'
    run_train_test(model, model_name, learning_rate, batch_size, 15)

