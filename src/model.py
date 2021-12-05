import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,ReLU,Sequential, Module
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
from PIL import Image
from time import time
from torchvision.models import resnet18, vgg11, efficientnet_b0, resnet152
from tqdm import tqdm_notebook

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#         x = self.softmax(x)
        return x


def construct_model(architecture, pretrained, device, out_features=10):
    if architecture == 'resnet18':
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=out_features, bias=True)
        print(model)
    elif architecture == 'resnet152':
        model = resnet152(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)
    elif architecture == 'vgg11':
        model = vgg11(pretrained = pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=out_features, bias=True)
    elif architecture == 'efficientnet_b0':
        model = efficientnet_b0(pretrained = pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features, bias=True)
    elif architecture == 'simplenet':
        model = Net()
    model = model.to(device)
    return model
