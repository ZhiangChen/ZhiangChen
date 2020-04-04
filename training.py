"""
training.py
Zhiang Chen, April 2020
"""

import torch
import torchvision.models as models
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

torch.manual_seed(0)

def neural_network(architecture, nm_classes, pretrained=True):
    assert architecture in model_names
    print("=> creating model '{}'".format(architecture))
    model = models.__dict__[architecture](pretrained=pretrained)
    if architecture.startswith('densenet'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features=in_features, out_features=nm_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=nm_classes)
    return model

def cifar10(root='./datasets/cifar10/', val=True):
    train = torchvision.datasets.CIFAR10(root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True)
    """
    if val:
        indices = torch.randperm(len(train)).tolist()
        train_set = torch.utils.data.Subset(train, indices[:-10000])
        val_set = torch.utils.data.Subset(train, indices[-10000:])
        return train_set, val_set, test
    """
    return train, test

def train():
    pass
