import torch
from torch import nn
import wandb
import sys

from dataloader import EICUDataSet
from dataloader import MIMICDataSet
from torch.utils.data import DataLoader

from models import BaseRecurrent, BaseLSTM, BaseGRU

from optimization import train_model, test_loop, val_loop
from evaluation import save_plot_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on device', device)

def get_dataloaders(config):
    if 'mimic' in config['dataset']:
        print('Loading MIMIC Dataset')
        train_data = MIMICDataSet('data/mimic/mimic_train_x.csv','data/mimic/mimic_train_y.csv')
        test_data = MIMICDataSet('data/mimic/mimic_test_x.csv','data/mimic/mimic_test_y.csv')
        val_data = MIMICDataSet('data/mimic/mimic_val_x.csv','data/mimic/mimic_val_y.csv')
    else:
        print('Loading eICU Dataset')
        train_data = EICUDataSet('data/eicu/eicu_train_x.csv','data/eicu/eicu_train_y.csv')
        test_data = EICUDataSet('data/eicu/eicu_test_x.csv','data/eicu/eicu_test_y.csv')
        val_data = EICUDataSet('data/eicu/eicu_val_x.csv','data/eicu/eicu_val_y.csv')

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader, val_loader

def get_model(config):
    if config.model_name == 'gru':
        model = BaseGRU(input_size=config.input_size, h_size=config.hidden_size, dropout=config.dropout).to(device)
    if config.model_name == 'lstm':
        model = BaseLSTM(input_size=config.input_size, h_size=config.hidden_size, dropout=config.dropout).to(device)
    if config.model_name == 'rnn':
        model = BaseRecurrent(input_size=config.input_size, h_size=config.hidden_size, dropout=config.dropout).to(device)
    return model

def run_train_test():
    run = wandb.init()
    config = wandb.config
    # wandb.run.name = config['model_name']+ '_' + wandb.run.id
    # wandb.run.save()

    print('Running training for '+config['model_name'])
    loss_fn =  nn.BCELoss()

    train_loader, test_loader, val_loader = get_dataloaders(config)
    model = get_model(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

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
            torch.save(model.state_dict(), 'models/'+config['model_name']+'.model')
            
        tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)
        wandb.log({'Train Loss': tr_l, 'Train Accuracy': tr_a}, commit=False)
        accurracies[0].append(tr_a)
        losses[0].append(tr_l)
        
        save_plot_loss(losses[0], losses[1], config['model_name'])

        # Early stopping
        if (max(losses[1][-2:]) == losses[1][-1]) and (len(losses[1]) > 3):
            maxes += 1
            print('Validation increased', maxes)
            if (maxes > 1):
                print('Stopping model training early', losses)
                break
        epoch += 1

        if epoch >= 300:
            print("Stopping model training early, max epochs")
            break

    model.load_state_dict(torch.load('models/'+config['model_name']+'.model'))
    test_loop(test_loader, model, loss_fn, config, wandb)
    run.finish()

if __name__ == '__main__':
    inputs = sys.argv
    if len(inputs) == 1:
        print('No sweep ID provided, cancelling')
    else:
        sweep_id = sys.argv[1]
        project_name = sys.argv[2]
        print('Running sweep for', sweep_id)
  
        wandb.agent(sweep_id, project=project_name, function=run_train_test)
    