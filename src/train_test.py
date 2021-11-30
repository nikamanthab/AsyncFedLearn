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
from tqdm import tqdm

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    y_pred = []
    y_true = []
    total_loss = 0.0
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        output = model(batch['image'].to(device))
        loss = criterion(output,batch['label'])
        total_loss += loss
        loss.backward()
        optimizer.step()
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy()) 
    acc = accuracy_score(y_pred, y_true)
    f1 = f1_score(y_pred, y_true, average='micro')

    return (acc, f1, total_loss, model)

def test(model, testloader, device):
    model.eval()
    y_pred = []
    y_true = []
    for batch in testloader:
        output = model(batch['image'].to(device))
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy())
    acc = accuracy_score(y_pred, y_true)
    f1 = f1_score(y_pred,y_true, average='micro')

    return (acc, f1)