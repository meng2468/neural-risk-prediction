import torch
from torch import nn
from sklearn.metrics import classification_report


def evaluate_pred(pred, y_true):
    y_pred = torch.argmax(pred, dim=1)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report
    
def train_model(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)

        loss = loss_fn(pred.view(-1, 2), y)
        total_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            # print(pred, y), loss
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    total_loss /= len(dataloader)
    correct /= size
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
            test_loss += loss_fn(pred.view(-1, 2), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            total_pred = torch.cat((total_pred, pred), 0)
            total_true += list(y)
    print('Eval deaths')
    print(evaluate_pred(total_pred, total_true)['0'])
    print('Eval survivors')
    print(evaluate_pred(total_pred, total_true)['1'])
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct