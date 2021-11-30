import sklearn
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,ReLU,Sequential, Module
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
from time import time
from tqdm import tqdm
from torchvision.models import resnet18, vgg11, alexnet
device = 'cuda'

label_dict = dict([(j,i) for (i,j) in list(enumerate(os.listdir('../Dataset/CIFAR-10-images/train/')))])
print(label_dict)

paths = []
label = []
for i in os.listdir('../Dataset/CIFAR-10-images/train/'):
    for j in os.listdir('../Dataset/CIFAR-10-images/train/'+str(i)):
        paths.append('../Dataset/CIFAR-10-images/train/'+str(i)+'/'+str(j))
        label.append(i)
train_df = pd.DataFrame({"paths": paths, "label": label})  

paths = []
label = []
for i in os.listdir('../Dataset/CIFAR-10-images/test/'):
    for j in os.listdir('../Dataset/CIFAR-10-images/test/'+str(i)):
        paths.append('../Dataset/CIFAR-10-images/test/'+str(i)+'/'+str(j))
        label.append(i)
test_df = pd.DataFrame({"paths": paths, "label": label})

print("train classes: ", len(train_df['label'].unique()))
print("train images: ", len(train_df['paths'].unique()))
print("test classes: ", len(test_df['label'].unique()))
print("test images: ", len(test_df['paths'].unique()))

train_df.to_csv('../Dataset/train.csv', index=False)
test_df.to_csv('../Dataset/test.csv', index=False)

class CIFARDataSet(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = csv_file
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
        trans = transforms.ToTensor()
        image = trans(image)
        sample = {
            "image": image.to(device),
            "label": torch.from_numpy(label_onehot).to(device)
        }
        return sample

model = vgg11(pretrained = True, progress = True)
model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
model = model.to(device)

train_batch_size = 64
learning_rate = 0.001
num_of_epochs = 20

# train_df = train_df.sample(frac=1)
traindataset = CIFARDataSet(train_df, transform=transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Resize((128,128)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)
trainloader = DataLoader(traindataset, batch_size=train_batch_size,shuffle=True)
valdataset = CIFARDataSet(test_df, transform=transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Resize((128,128)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
valloader = DataLoader(valdataset, batch_size=32)

criterian = CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

train_f1 = []
test_f1 = []
print("----------------starting train loop------------------")
for epoch in range(num_of_epochs):
    start_time = time()
    print("epoch:",epoch)
    y_pred = []
    y_true = []
    total_loss = 0.0
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        output = model(batch['image'].to(device))
        loss = criterian(output,batch['label'])
        total_loss += loss
        loss.backward()
        optimizer.step()
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy()) 
    print("training acc:",accuracy_score(y_pred,y_true),end=' ')
    f1 = f1_score(y_pred,y_true, average='micro')
    train_f1.append(f1)
    print("total loss:", total_loss, end=' ')
    print("training f1_score:", f1)
    end_time = time()
    print("Time for one epoch:", end_time - start_time)

    print("samples/sec:", len(train_df)/(end_time - start_time))
    
    y_pred = []
    y_true = []
    for batch in valloader:
        output = model(batch['image'].to(device))
        y_pred += list(output.argmax(dim=1).detach().cpu().numpy())
        y_true += list(batch['label'].argmax(dim=1).detach().cpu().numpy())
    print("test acc:",accuracy_score(y_pred,y_true),end=' ')
    f1 = f1_score(y_pred,y_true, average='micro')
    test_f1.append(f1)
    print("test f1_score:", f1)

