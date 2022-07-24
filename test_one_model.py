import os

import torch
import torch.nn as nn
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import tqdm
from model import MLP, weights_init_normal, CNN


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Training data loader
    train_loader = torch.utils.data.DataLoader(
        FashionMNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        FashionMNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=64, shuffle=True)

    model = MLP()
    model.apply(weights_init_normal)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    logname = "logs/mlp"
    if not os.path.exists(logname):
        os.makedirs(logname, exist_ok=True)
    writer = SummaryWriter(logname)
    for epoch in range(10):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for step, (x, y) in enumerate(train_loader):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                out = model(x)
                loss = loss_func(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), step + epoch * len(train_loader))
                pbar.set_description('Epoch: %d, Step: %d, Loss: %.4f' % (epoch, step, loss.item()))
                pbar.update(1)
        # calculate accuracy of the model
        correct = 0
        total = 0
        for (x, y) in test_loader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()
        print('Accuracy of the model on the test images: %f %%' % (correct / total))
        writer.add_scalar('accuracy', correct / total, epoch)


if __name__ == "__main__":
    main()
# MLP 88.27%
# CNN 85.32%
