import torch
from torch import nn
import sys

from dataloader import EICUDataSet
from torch.utils.data import DataLoader
from eicu_models import BaseRecurrent, LayeredRecurrent
from eicu_models import BaseLSTM, BaseGRU

from optimization import train_model, test_loop, val_loop
from evaluation import save_plot_loss


def run_train_test(model, model_name, learning_rate, batch_size):
    print('Running training for '+model_name)
    loss_fn =  nn.BCELoss()

    train_data = EICUDataSet('data/eicu/eicu_train_x.csv','data/eicu/eicu_train_y.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = EICUDataSet('data/eicu/eicu_test_x.csv','data/eicu/eicu_test_y.csv')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    val_data = EICUDataSet('data/eicu/eicu_val_x.csv','data/eicu/eicu_val_y.csv')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = [[],[]]
    accurracies = [[],[]]
    maxes = 0
    epoch = 0
    while True:
        print('*'*20)
        print('Running Epoch', epoch)
        
        val_l, val_a = val_loop(val_loader, model, loss_fn, model_name, epoch)
        losses[1].append(val_l)
        accurracies[1].append(val_a)

        if val_l == min(losses[1]):
            print('Best val model performance, storing')
            torch.save(model.state_dict(), 'models/'+model_name+'.model')
        
        tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)
        losses[0].append(tr_l)
        accurracies[0].append(tr_a)
        
        save_plot_loss(losses[0], losses[1], model_name)

        # Early stopping
        if (max(losses[1][-2:]) == losses[1][-1]) and (len(losses[1]) > 3):
            maxes += 1
            print('Validation increased', maxes)
            if maxes > 1:
                print('Stopping model training early', losses)
                break
        epoch += 1
        
        if epoch >= 15:
            break

    model.load_state_dict(torch.load('models/'+model_name+'.model'))
    test_loop(test_loader, model, loss_fn, model_name, epoch)
    
if __name__ == '__main__':
    learning_rates = [1e-2,5e-3,1e-3,5e-4,1e-4,5e-5]
    # learning_rate = 1e-3
    batch_size = 50
    
    for learning_rate in learning_rates:
        model = BaseGRU()
        model_name = 'eicu_base_gru'+str(learning_rate)
        run_train_test(model, model_name, learning_rate, batch_size)

        # model = LayeredRecurrent()
        # model_name = 'eicu_layered_rnn'
        # run_train_test(model, model_name, learning_rate, batch_size)
        
        model = BaseLSTM()
        model_name = 'eicu_base_lstm'+str(learning_rate)
        run_train_test(model, model_name, learning_rate, batch_size)

        model = BaseRecurrent()
        model_name = 'eicu_base_rnn'+str(learning_rate)
        run_train_test(model, model_name, learning_rate, batch_size)

