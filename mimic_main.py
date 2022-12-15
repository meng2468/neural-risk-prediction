import torch
from torch import nn
import sys

from dataloader import MIMICDataSet
from torch.utils.data import DataLoader
from mimic_models import BaseRecurrent, LayeredRecurrent
from mimic_models import BaseLSTM, BaseGRU

from optimization import train_model, test_loop
from evaluation import save_plot_loss

def run_train_test(model, model_name, learning_rate, batch_size):
    print('Running training for '+model_name)
    loss_fn =  nn.BCELoss()

    train_data = MIMICDataSet('data/mimic/mimic_train_x.csv','data/mimic/mimic_train_y.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # test_data = MIMICDataSet('data/eicu/eicu_test_x.csv','data/eicu/eicu_test_y.csv')
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    val_data = MIMICDataSet('data/mimic/mimic_val_x.csv','data/mimic/mimic_val_y.csv')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = [[],[]]
    accurracies = [[],[]]
    maxes = 0
    epoch = 0
    while True:
        print('*'*20)
        print('Running Epoch', epoch)
        
        ts_l, ts_a = test_loop(val_loader, model, loss_fn, model_name, epoch)
        tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)

        losses[0].append(tr_l)
        losses[1].append(ts_l)

        accurracies[0].append(tr_a)
        accurracies[1].append(ts_a)
        
        save_plot_loss(losses[0], losses[1], model_name)

        # Early stopping
        if max(losses[1][-3:]) == losses[1][-1] and len(losses[1]) > 3:
            maxes += 1
            if maxes > 3:
                print('Stopping model training early', losses)
                break
        epoch += 1
        

    torch.save(model.state_dict, 'models/'+model_name+'.model')
    

if __name__ == '__main__':
    learning_rate = 1e-5
    batch_size = 50
    
    model = BaseRecurrent()
    model_name = 'mimic_base_rnn'
    run_train_test(model, model_name, learning_rate, batch_size)
    
    model = BaseLSTM()
    model_name = 'mimic_base_lstm'
    run_train_test(model, model_name, learning_rate, batch_size)

    model = BaseGRU()
    model_name = 'mimic_base_gru'
    run_train_test(model, model_name, learning_rate, batch_size)

    model = LayeredRecurrent()
    model_name = 'mimic_layered_rnn'
    run_train_test(model, model_name, learning_rate, batch_size)

