import torch
from torch import nn
import sys

from dataloader import MIMICDataSet
from torch.utils.data import DataLoader
from mimic_models import BaseRecurrent, LayeredRecurrent
from mimic_models import BaseLSTM, BaseGRU

from optimization import train_model, val_loop, test_loop
from evaluation import save_plot_loss

import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on device', device)

def run_train_test(model, params, experiment_name):
    run = wandb.init(project=experiment_name, entity="risk-prediction", config=params, reinit=True)
    wandb.run.name = params['model_name']+ '_' + wandb.run.id
    wandb.run.save()

    print('Running training for '+params['model_name'])
    loss_fn =  nn.BCELoss()

    train_data = MIMICDataSet('data/mimic/mimic_train_x.csv','data/mimic/mimic_train_y.csv')
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    test_data = MIMICDataSet('data/mimic/mimic_test_x.csv','data/mimic/mimic_test_y.csv')
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    val_data = MIMICDataSet('data/mimic/mimic_val_x.csv','data/mimic/mimic_val_y.csv')
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    losses = [[],[]]
    accurracies = [[],[]]
    maxes = 0
    epoch = 0

    while True:
        print('*'*20)
        print('Running Epoch', epoch)
        tr_l, tr_a = -1, -1
        val_l, val_a = val_loop(val_loader, model, loss_fn)
        wandb.log({'Val Loss': val_l, 'Val Accuracy': val_a})
        accurracies[1].append(val_a)
        losses[1].append(val_l)
        
        if val_l == min(losses[1]):
            print('Best val model performance, storing')
            torch.save(model.state_dict(), 'models/'+params['model_name']+'.model')
            
        tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)
        wandb.log({'Train Loss': tr_l, 'Train Accuracy': tr_a}, commit=False)
        accurracies[0].append(tr_a)
        losses[0].append(tr_l)
        
        save_plot_loss(losses[0], losses[1], params['model_name'])

        # Early stopping
        if (max(losses[1][-2:]) == losses[1][-1]) and (len(losses[1]) > 3):
            maxes += 1
            print('Validation increased', maxes)
            if (maxes > 1):
                print('Stopping model training early', losses)
                break
        epoch += 1

        if epoch >= 40:
            break

    model.load_state_dict(torch.load('models/'+params['model_name']+'.model'))
    test_loop(test_loader, model, loss_fn, params, wandb)
    run.finish()

if __name__ == '__main__':
    experiment_name = 'mimic-uncapped-epochs'
    learning_rates = [5e-4,1e-4,5e-5, 1e-5]
    batch_size = 50

    for i in range(5):
        for learning_rate in learning_rates:
            params = {'learning_rate': learning_rate, 'batch_size': batch_size}
            params['iteration'] = i

            model = BaseGRU().to(device)
            params['model_name'] = 'mimic_base_gru'
            run_train_test(model, params, experiment_name)
            
            model = BaseLSTM().to(device)
            params['model_name'] = 'mimic_base_lstm'
            run_train_test(model, params, experiment_name)

            model = BaseRecurrent().to(device)
            params['model_name'] = 'mimic_base_rnn'     
            run_train_test(model, params, experiment_name)


