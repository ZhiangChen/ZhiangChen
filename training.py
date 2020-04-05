"""
training.py
Zhiang Chen, April 2020
"""

import torch
import torch.utils.data
import torchvision.datasets
import torch.nn as nn
import torchvision.transforms as transforms
from utils import *
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

torch.manual_seed(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,])

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize,])

def neural_network(architecture, nm_classes, pretrained=True, change_last_layer=False):
    assert architecture in model_names
    print("=> creating model '{}'".format(architecture))
    model = models.__dict__[architecture](pretrained=pretrained)
    if change_last_layer:
        if architecture.startswith('densenet'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=in_features, out_features=nm_classes)
        else:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=nm_classes)

    return model

def cifar10(root='./datasets/cifar10/', val=True):
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
    """
    if val:
        indices = torch.randperm(len(train)).tolist()
        train_set = torch.utils.data.Subset(train, indices[:-10000])
        val_set = torch.utils.data.Subset(train, indices[-10000:])
        return train_set, val_set, test
    """
    return train, test




if __name__ == '__main__':
    cuda = 'cuda:0'
    device = torch.device(cuda)
    nm_classes = 10
    train_dataset, test_dataset = cifar10()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = neural_network('resnet152', nm_classes)

    #if you want to load weight

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    init_epoch = 0
    num_epochs = 2
    print_freq = 500

    save_param = "trained_param_cifar10/epoch_{:04d}.param".format(init_epoch)
    torch.save(model.state_dict(), save_param)

    for epoch in range(init_epoch, init_epoch + num_epochs):
        save_param = "trained_param_cifar10/epoch_{:04d}.param".format(epoch)
        train(train_dataloader, model, criterion, optimizer, epoch, device, print_freq)
        lr_scheduler.step()
        validate(test_dataloader, model, criterion, device)
        torch.save(model.state_dict(), save_param)
