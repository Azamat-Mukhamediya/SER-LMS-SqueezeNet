import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.preprocessing import SERDataset


def get_mean_and_std(segmented_train, batch_size):

    transform_temp = transforms.Compose([transforms.ToTensor()
                                         ])
    train_dataset_temp = SERDataset(
        segmented_train, 'train', transform=transform_temp)

    dataloader = DataLoader(
        train_dataset_temp, batch_size=batch_size, shuffle=True, num_workers=0)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X_device, y_device = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X_device)
        loss = loss_fn(pred, y_device)

        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y_device).type(torch.float).sum().item()

        if batch % 16 == 15:

            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        del X_device
        del y_device

    train_loss /= num_batches
    correct /= size
    train_acc = 100*correct

    return train_loss, train_acc


def evaluate(dataloader, model, loss_fn, device, type):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X_device, y_device = X.to(device), y.to(device)
            pred = model(X_device)
            test_loss += loss_fn(pred, y_device).item()
            correct += (pred.argmax(1) ==
                        y_device).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"{type} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_acc = 100*correct
    return test_loss, test_acc
