import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

from gentic import model_list, list_model_initial, list_model_train, list_model_test_accuracy, reproduce, mute
from model import MLP, CNN

import numpy as np

epoch_num = 40
batch_size = 64
lr = 0.001
partical_num = 10
model_type = "mlp"

best_model = MLP() if model_type == "mlp" else CNN()
best_accuracy = 0
# initial random seed
random.seed(0)
torch.manual_seed(0)

cuda = torch.cuda.is_available()


def main():
    models_list = model_list(partical_num, model_type)
    models_list = list_model_initial(models_list, True, cuda)
    # mnist dataloader
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    # optimizer
    optimizer_list = [torch.optim.Adam(model.parameters(), lr=lr) for model in models_list]
    logname = './logs/gen-40epochs'
    if not os.path.exists(logname):
        os.makedirs(logname, exist_ok=True)
    writer = SummaryWriter(logname)
    for epoch in range(epoch_num):
        print("\n====================Epoch {}====================".format(epoch))
        # train
        models_list = list_model_train(models_list, train_loader, optimizer_list, cuda)
        # test
        accuracy_list = list_model_test_accuracy(models_list, test_loader, cuda)
        # reproduce
        child = reproduce(models_list, accuracy_list=accuracy_list, parents_number=2, model_type=model_type,
                          mode="average")
        # mute
        if epoch == epoch_num - 1:
            global best_model, best_accuracy
            best_model = models_list[np.argmax(accuracy_list)]
            best_accuracy = np.max(accuracy_list)
        writer.add_scalar('accuracy', np.max(accuracy_list), epoch)
        # mute
        models_list = mute(child, num_paricles=10, mute_rate=0.001, noise_var=0.01, model_type=model_type)
        # re-initialize the models
        models_list = list_model_initial(models_list, init_weight=False, cuda=cuda)  # init_weight=False
        # restart optimizer
        optimizer_list = [torch.optim.Adam(model.parameters(), lr=lr) for model in models_list]
    if not os.path.exists("model"):
        os.makedirs("model")
    print("Best accuracy: {}".format(best_accuracy))
    print("Save best model to model/best_model.pth")
    torch.save(best_model.state_dict(), './model/best_model.pth')


if __name__ == '__main__':
    main()
# MLP: 88.63%
