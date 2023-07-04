import torch
from torch import nn
import sys

from dataloader import EICUDataSet
from dataloader import MIMICDataSet
from torch.utils.data import DataLoader

from models import BaseRecurrent, BaseLSTM, BaseGRU

from optimization import train_model, test_loop, val_loop
from evaluation import save_plot_loss


from sklearn.metrics import classification_report, roc_auc_score
import csv

import os
import torch
from torch import nn

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

device='cuda:1'

class BaseRecurrent(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseRecurrent, self).__init__()
        self.hidden_size = h_size
        self.recurrent = nn.RNN(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size,out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.recurrent(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits
    
class BaseLSTM(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseLSTM, self).__init__()
        self.hidden_size = h_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits
    
class BaseGRU(nn.Module):
    def __init__(self, input_size, h_size, dropout):
        super(BaseGRU, self).__init__()
        self.hidden_size = h_size
        self.lstm = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, dropout=dropout)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        final = self.final(out[:,-1,:])
        logits = self.sigmoid(final)
        return logits

config = {
    'dataset': 'mimic',
    'batch_size': 50,
}

# Dataset Loading

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

mimic_configs = [{'dataset': 'mimic',
  'model_name': 'lstm',
  'batch_size': 50,
  'input_size': 51,
  'learning_rate': 0.0006117734540819,
  'dropout': 0.377348671036311,
  'hidden_size': 4096},
 {'dataset': 'mimic',
  'model_name': 'rnn',
  'batch_size': 50,
  'input_size': 51,
  'learning_rate': 0.0005402984097166,
  'dropout': 0.0750560926214659,
  'hidden_size': 512},
 {'dataset': 'mimic',
  'model_name': 'gru',
  'batch_size': 50,
  'input_size': 51,
  'learning_rate': 0.0001780968308659,
  'dropout': 0.0553891378020515,
  'hidden_size': 4096}]

eicu_configs = [{'dataset': 'eicu',
  'model_name': 'lstm',
  'batch_size': 50,
  'input_size': 45,
  'learning_rate': 0.000273251376771,
  'dropout': 0.1095184746793898,
  'hidden_size': 2048},
 {'dataset': 'eicu',
  'model_name': 'rnn',
  'batch_size': 50,
  'input_size': 45,
  'learning_rate': 0.000140241753481,
  'dropout': 0.096768530504191,
  'hidden_size': 1024},
 {'dataset': 'eicu',
  'model_name': 'gru',
  'batch_size': 50,
  'input_size': 45,
  'learning_rate': 5.68735719201e-05,
  'dropout': 0.1396662997746955,
  'hidden_size': 8192}]

# Model Loading
for i in range(5):
    print('*'*30)
    print("Running iteration", i)
    for model_config in mimic_configs:
        for k, v in model_config.items():
            print('Setting', k, 'to', v)
            config[k] = v

        if config['model_name'] == 'gru':
            model = BaseGRU(input_size=config['input_size'], h_size=config['hidden_size'], dropout=config['dropout']).to(device)
        if config['model_name'] == 'lstm':
            model = BaseLSTM(input_size=config['input_size'], h_size=config['hidden_size'], dropout=config['dropout']).to(device)
        if config['model_name'] == 'rnn':
            model = BaseRecurrent(input_size=config['input_size'], h_size=config['hidden_size'], dropout=config['dropout']).to(device)

        loss_fn =  nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        losses = [[],[]]
        accurracies = [[],[]]
        maxes = 0
        epoch = 0

        model_file_path = 'final_models/'+config['model_name']+'_'+config['dataset']+'_'+str(i)+'.model'
        model_log_path = 'final_models/'+config['model_name']+'_'+config['dataset']+'_'+str(i)+'.csv'
        
        with open(model_log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['tr_loss','val_loss','tr_acc','val_acc'])
        print('Model with file path', model_file_path)
        while True:
            print('*'*20)
            print('Running Epoch', epoch)
            tr_l, tr_a = -1, -1
            val_l, val_a = val_loop(val_loader, model, loss_fn)
            
            accurracies[1].append(val_a)
            losses[1].append(val_l)

            if val_l == min(losses[1]):
                print('Best val model performance, storing')
                torch.save(model.state_dict(), model_file_path)

            tr_l, tr_a = train_model(train_loader, model, loss_fn, optimizer)
            print({'Val Loss': val_l, 'Val Accuracy': val_a})

            print({'Train Loss': tr_l, 'Train Accuracy': tr_a})

            with open(model_log_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([tr_l, val_l,tr_a,val_a])

            accurracies[0].append(tr_a)
            losses[0].append(tr_l)
            
        #     save_plot_loss(losses[0], losses[1], config['model_name'])

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

        model.load_state_dict(torch.load(model_file_path))
        val_loss, _ = val_loop(val_loader, model, loss_fn)
        print({'Final Val Loss': val_loss})


        dataloader = test_loader

        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        total_pred = torch.tensor([]).to(device)
        total_true = []

        with torch.no_grad():
            for X, y in dataloader:    
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (torch.round(pred) == y).sum()

                total_pred = torch.cat((total_pred, pred), 0)
                total_true += list(y)

        y_pred = torch.round(total_pred)

        y_true = [int(x) for x in total_true]
        pred = [int(x) for x in y_pred]
        report = classification_report(y_true, pred, output_dict=True)
        roc_auc = roc_auc_score(y_true, pred)

        f1 = report

        # Store model performance in a csv
        experiment_file = 'final_experiments.csv'
        if experiment_file not in os.listdir('evaluation/'):
            with open(experiment_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list(config.keys()) + ['precision','recall','f1-score','support','roc auc','accuracy'])

        with open(experiment_file, 'a') as f:
            writer = csv.writer(f)
            store = [v for _, v in config.items()]
            store += [y for _, y in report['macro avg'].items()]
            store += [roc_auc]+[report['accuracy']]
            writer.writerow(store)

        print(dict(list(report['macro avg'].items()) + list({'roc auc': roc_auc, 'accuracy': report['accuracy']}.items())))
        print('-'*20)
        print('Running Test Evaluation')
        for k, v in f1.items():
            print(k, v)
            print('')
        print('ROC AUC:',roc_auc)
        print('-'*20)

        test_loss /= num_batches
        correct = int(correct)/size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        print(' finished training with ROC ' + str(roc_auc) + ' and loss of ' + str(test_loss))

        print(f1['macro avg']['f1-score'], roc_auc)



