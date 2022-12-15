import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_pred(pred, y_true):
    y_true = [int(x) for x in y_true]
    pred = [int(x) for x in pred]
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_true, pred)
    return report, roc_auc
    
def train_model(dataloader, model, loss_fn, optimizer):
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

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    total_pred = torch.tensor([])
    total_true = []

    with torch.no_grad():
        for X, y in dataloader:    
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum()

            total_pred = torch.cat((total_pred, pred), 0)
            total_true += list(y)

    y_pred = torch.round(total_pred)

    f1, roc_auc = evaluate_pred(y_pred, total_true)
    print('')
    print('-'*20)
    print('Evaluation')
    for k, v in f1.items():
        print(k, v)
    print('ROC AUC:',roc_auc)
    print('-'*20)

    test_loss /= num_batches
    correct = int(correct)/size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

def evaluate_model(dataloader, model, loss_fn):
    total_pred = torch.tensor([])
    total_true = []

    with torch.no_grad():
        for X, y in dataloader:    
            pred = model(X)
            test_loss += loss_fn(pred.view(-1, 2), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            total_pred = torch.cat((total_pred, pred), 0)
            total_true += list(y)

    return evaluate_pred(total_pred, total_true)