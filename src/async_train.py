import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,ReLU,Sequential, Module
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
from time import time
from tqdm import tqdm
from data_loader import get_test_dataloader, get_train_dataloader
from generate_iid import generate_test_data, generate_train_data
from train_test import train ,test
from model import construct_model
import copy
from aggregator import fed_avg_aggregator, comed_aggregator


model_list = []
number_of_samples = [20, 50, 30]
device = 'cuda:0'
architecture ='simplenet'
batch_size = 32
number_of_iterations = 50
number_of_epochs = 1
learning_rate = 0.01
pretrained = True
aggregator = 'fedavg' #fedavg or comed
out_features = 10

#create dfs
traindf_list = generate_train_data(number_of_samples)
test_df = generate_test_data()

#generate dataloaders
trainloaders_list = []
for df in traindf_list:
    trainloaders_list.append(get_train_dataloader(df, batch_size))

testloader = get_test_dataloader(test_df, batch_size)

#initialize common model
init_model = construct_model(architecture, pretrained, device, out_features)
model_list = []
for idx in range(len(number_of_samples)):
    model_list.append([number_of_samples[idx], copy.deepcopy(init_model)])

#run loop and run all models
for iter in range(1, number_of_iterations+1):
    print("iter: ", str(iter))
    for node_idx in range(len(number_of_samples)):
        print("node: ", node_idx)
        model = model_list[node_idx][1]
        trainloader = trainloaders_list[node_idx]
        criterion = CrossEntropyLoss().to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        for epochs in range(number_of_epochs):
            acc, f1, total_loss = train(model, trainloader, optimizer, criterion, device)
            print("\tTraining acc:", acc, end=' ')
            print("| loss:", total_loss.item(), end=' ')
            print("| F1:", f1, end=' ')
            print("||", end=' ')

            #test at each step
            acc, f1 = test(model, testloader, device)
            print("Test acc:", acc, end=' ')
            print("| F1:", f1)
        model_list[node_idx][1] = model

    if aggregator == 'fedavg':
        agg_model = fed_avg_aggregator(model_list, architecture, pretrained, device, out_features)
    elif aggregator == 'comed':
        agg_model = comed_aggregator(model_list, architecture, pretrained, out_features)
    
    acc, f1 = test(agg_model, testloader, device)
    print("Test acc:", acc, end=' ')
    print("| F1:", f1)
    
    for idx in range(len(model_list)):
        model_list[idx][1] = agg_model