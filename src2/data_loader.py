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
from torchvision.models import resnet18, alexnet, vgg11, efficientnet_b0, resnet152
from tqdm import tqdm_notebook

label_dict = dict([(j,i) for (i,j) in sorted(list(enumerate(os.listdir('../Dataset/CIFAR-10-images/train/'))))])

class CIFARDataSet(Dataset):
    def __init__(self, csv_file, transform=None, device='cuda'):
        self.transform = transform
        self.df = csv_file
        self.device = device
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = label_dict[self.df.iloc[idx]['label']]
        label_onehot = np.zeros(10)
        label_onehot[label] = 1.
        path = self.df.iloc[idx]['paths']
        image = Image.open(path)
        image = self.transform(image)
        sample = {
            "image": image.to(self.device),
            "label": torch.from_numpy(label_onehot).to(self.device)
        }
        return sample

def get_train_dataloader(train_df, train_batch_size=32, device='cuda'):
    traindataset = CIFARDataSet(train_df, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), device=device)
    trainloader = DataLoader(traindataset, batch_size=train_batch_size,shuffle=True)
    return trainloader

def get_test_dataloader(test_df='../Dataset/test.csv', device='cuda'):
    valdataset = CIFARDataSet(test_df, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), device=device)
    valloader = DataLoader(valdataset, batch_size=5000)
    return valloader

