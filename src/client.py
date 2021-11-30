import requests
import os
import time
import pandas as pd
import torch
from data_loader import get_test_dataloader, get_train_dataloader
from generate_iid import generate_test_data, generate_train_data
from train_test import train ,test
from model import construct_model
import copy
from aggregator import fed_avg_aggregator, comed_aggregator
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,Softmax,CrossEntropyLoss,ReLU,Sequential, Module
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Client module.')
parser.add_argument('--number', type=str, default='0', \
    help='0, 1, 2, ...')
args = parser.parse_args()

node_details = {
    'node_number': args.number,
    'local_init_counter': 0
}

def getConnection(url, node_details):
    return requests.post(url+'/getConnection', json=node_details).json()

def getModel(url, path):
    while(True):
        response = requests.post(url+'/checkphase', \
            json={} \
                )
        server_phase = response.json()['phase']
        print(server_phase, ",", node_details['local_init_counter'])
        if server_phase != node_details['local_init_counter']:
            print('waiting...')
            time.sleep(10)
            continue
        else:
            break
    response = requests.post(url+'/getmodel', stream=True)
    model_file = open(path,"wb")
    for chunk in response.iter_content(chunk_size=1024):
        model_file.write(chunk)
    model_file.close()
    node_details['local_init_counter']+=1

def sendModel(url, path, node_details):
    res = requests.post(url+'/sendmodel', files={'file': ('node_'+node_details['node_number']+'.pt', open(path, 'rb'))}, stream=True)
    if res.json()['status'] == 'doaggregation':
        final_res = requests.post(url+'/doaggregation')
    print(res.json()['status'])

##########################

# Get connection
server_params = getConnection('http://127.0.0.1:5000', node_details)
node_details.update(server_params)

#Load dataframe
train_df = pd.read_csv('./dataframes/node_'+str(node_details['node_number'])+'.csv')
test_df = pd.read_csv('./dataframes/test.csv')

trainloader = get_train_dataloader(train_df, node_details['batch_size'], node_details['device'])
testloader = get_test_dataloader(test_df, node_details['device'])

while(True):

    # Get init or agg model
    getModel('http://127.0.0.1:5000', './client_models/node_'+str(node_details['node_number'])+'.pt')
    model = torch.load('./client_models/node_'+str(node_details['node_number'])+'.pt').to(node_details['device'])
    model = model.to(node_details['device'])

    # creating optimizer and loss fn
    criterion = CrossEntropyLoss().to(node_details['device'])
    optimizer = SGD(model.parameters(), lr=node_details['learning_rate'])

    #training loop
    for epochs in range(node_details['number_of_epochs']):
        print(node_details['device'])
        acc, f1, total_loss, model = train(model, trainloader, optimizer, criterion, node_details['device'])
        print("\tTraining acc:", acc, end=' ')
        print("| loss:", total_loss.item(), end=' ')
        print("| F1:", f1, end=' ')
        print("||", end=' ')

        #test at each step
        acc, f1 = test(model, testloader, node_details['device'])
        print("Test acc:", acc, end=' ')
        print("| F1:", f1)
    torch.save(model, 'client_models/node_'+str(node_details['node_number'])+'.pt')
    sendModel('http://127.0.0.1:5000', 'client_models/node_'+str(node_details['node_number'])+'.pt', node_details)

