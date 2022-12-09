import torch
from torch import nn

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
    return total_loss, correct

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.view(-1, 2), y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct