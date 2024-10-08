import os
import csv

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, roc_auc_score

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def evaluate_pred(pred, y_true, config, wandb):
    probs = pred.cpu()
    pred = torch.round(pred)
    y_true = [int(x) for x in y_true]
    pred = [int(x) for x in pred]

    roc_auc = roc_auc_score(y_true, probs)
    report = classification_report(y_true, pred, output_dict=True)

    # Store model performance in a csv
    if 'experiments.csv' not in os.listdir('evaluation/'):
        with open('evaluation/experiments.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list(config.keys()) + ['precision','recall','f1-score','support','roc auc','accuracy'])


    with open('evaluation/experiments.csv', 'a') as f:
        writer = csv.writer(f)
        store = [v for _, v in config.items()]
        store += [y for _, y in report['macro avg'].items()]
        store += [roc_auc]+[report['accuracy']]
        writer.writerow(store)

    wandb.log(dict(list(report['macro avg'].items()) + list({'roc auc': roc_auc, 'accuracy': report['accuracy']}.items())))
    return report, roc_auc
    
def train_model(dataloader, model, loss_fn, optimizer):
    model.train()
    size=len(dataloader.dataset)
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)

        loss = loss_fn(pred, y)
        total_loss += loss
        correct += (torch.round(pred) == y).sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            # print(pred, y), loss
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    total_loss /= len(dataloader)
    correct = int(correct)/size

    print(f"Train  Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss.item(), correct

def val_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    total_pred = torch.tensor([]).to(device)
    total_true = []

    with torch.no_grad():
        for X, y in dataloader:    
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum()

            total_pred = torch.cat((total_pred, pred), 0)
            total_true += list(y)
    
    val_loss /= num_batches
    correct = int(correct)/size
    print('')
    print(f"Validation Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss, correct

def test_loop(dataloader, model, loss_fn, config, wandb):
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


    f1, roc_auc = evaluate_pred(total_pred, total_true, config, wandb)

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

    wandb.alert(
        title='Model Completed Evaluation',
        text=config['model_name'] + ' finished training with ROC ' + str(roc_auc) + ' and loss of ' + str(test_loss)
    )
    return f1['macro avg']['f1-score'], roc_auc
