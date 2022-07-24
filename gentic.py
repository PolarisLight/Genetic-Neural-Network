import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import tqdm
import random

from model import MLP, weights_init_normal, CNN


def model_list(num_particals=10, model_type="MLP"):
    model_type = model_type.upper()
    assert model_type in ["MLP", "CNN"]
    model_list = []
    if model_type == "MLP":
        for i in range(num_particals):
            model_list.append(MLP())
    elif model_type == "CNN":
        for i in range(num_particals):
            model_list.append(CNN())
    return model_list


def list_model_initial(model_list, init_weight=True, cuda=False):
    if init_weight == True:
        for model in model_list:
            model.apply(weights_init_normal)
    if cuda:
        for model in model_list:
            model.cuda()
    return model_list


def list_model_train(model_list, train_loader, optimizer_list, cuda=False, writer=None, n_epochs=0):
    all_model_loss = []
    with tqdm.tqdm(total=len(model_list)) as pbar:
        for i, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
            loss_list = []
            model.train()
            for j,(data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = F.cross_entropy(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if writer is not None:
                    writer.add_scalar("model {} loss".format(i), loss.item(), n_epochs * len(train_loader) + j)
                loss_list.append(loss.item())
            all_model_loss.append(np.mean(loss_list))
            pbar.set_description("Model {}: {:.4f}".format(i, all_model_loss[-1]))
            pbar.update(1)
    # print the loss of each model
    for i, loss in enumerate(all_model_loss):
        print("Model {}: {:.4f}".format(i, loss), end=" ")
    print()
    return model_list


def list_model_test_accuracy(model_list, test_loader, cuda=False,writer=None, n_epochs=0):
    accuracy_list = []
    for model in model_list:
        model.eval()
        test_accuracy = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()
        accuracy_list.append(test_accuracy / len(test_loader.dataset))
        if writer is not None:
            writer.add_scalar("model {} accuracy".format(model_list.index(model)), accuracy_list[-1], n_epochs)
    # print the accuracy of each model
    print("Accuracy: ", )
    for i, accuracy in enumerate(accuracy_list):
        print("Model {}: {:.4f}".format(i, accuracy), end=" ")
    print()
    return accuracy_list


def reproduce(model_list, accuracy_list, parents_number=2, mode="average", model_type="MLP"):
    assert mode in ["average", "concat"]
    model_type = model_type.upper()
    assert model_type in ["MLP", "CNN"]
    accuracy_array = np.array(accuracy_list)
    a = accuracy_array[np.argpartition(accuracy_array, -parents_number)[-parents_number:]]
    parent1 = model_list[accuracy_list.index(a[0])]
    parent2 = model_list[accuracy_list.index(a[1])]
    print(f"Parent1: Model {accuracy_list.index(a[0])},Parent2: Model {accuracy_list.index(a[1])}")
    dic1 = parent1.state_dict()
    dic2 = parent2.state_dict()
    for layer in dic1.keys():
        if layer.find("weight") != -1:
            if mode == "average":
                dic1[layer] = (dic1[layer] + dic2[layer]) / 2
            elif mode == "concat":
                dic1[layer] = torch.cat(
                    (dic1[layer][:dic1[layer].shape[0] // 2], dic2[layer][dic2[layer].shape[0] // 2:]), 0)
        elif layer.find("bias") != -1:
            if mode == "average":
                dic1[layer] = (dic1[layer] + dic2[layer]) / 2
            elif mode == "concat":
                dic1[layer] = torch.cat(
                    (dic1[layer][:dic1[layer].shape[0] // 2], dic2[layer][dic2[layer].shape[0] // 2:]), 0)
    if model_type == "MLP":
        child = MLP()
    elif model_type == "CNN":
        child = CNN()
    child.load_state_dict(dic1)
    return child


def mute(child, num_paricles, mute_rate=0.01, noise_var=0.01, model_type="MLP"):
    list = model_list(num_paricles, model_type)
    dic = child.state_dict()

    for model in list:
        temp_dic = dic.copy()
        for layer in dic.keys():
            if layer.find("weight") != -1:
                if random.randint(0, 100) > int(mute_rate * 100):
                    temp_dic[layer] = temp_dic[layer] + torch.randn(temp_dic[layer].shape) * noise_var
                else:
                    temp_dic[layer] = temp_dic[layer] + torch.randn(temp_dic[layer].shape) * noise_var * 10
            elif layer.find("bias") != -1:
                if random.randint(0, 100) > int(mute_rate * 100):
                    dic[layer] = dic[layer] + torch.randn(dic[layer].shape) * noise_var
                else:
                    dic[layer] = dic[layer] + torch.randn(dic[layer].shape) * noise_var * 10
        model.load_state_dict(temp_dic)
    return list


if __name__ == "__main__":
    models_list = model_list("cnn", 2)
    acc_list = [10, 20]
    child = reproduce(models_list, accuracy_list=acc_list, parents_number=2, mode="average", model_type="CNN")
