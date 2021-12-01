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
import threading
import os
from aggregator import fed_avg_aggregator, comed_aggregator

from flask import Flask, request, send_from_directory, send_file
import requests
import json
app = Flask(__name__)

for i in os.listdir('models'):
    os.remove('models/'+i)

params = {
    'node_names': [],
    'number_of_samples' : [30, 70],
    'device' : 'cpu',
    'architecture' : 'simplenet',
    'batch_size' : 2,
    'number_of_iterations' : 50,
    'number_of_epochs' : 1,
    'learning_rate' : 0.01,
    'pretrained' : True,
    'aggregator' : 'fedavg', #fedavg or comed
    'out_features' : 10
}

#create dfs
traindf_list = generate_train_data(params['number_of_samples'])
test_df = generate_test_data()
test_df.to_csv('./dataframes/test.csv', index=False)
testloader = get_test_dataloader(test_df)

for idx, df in enumerate(traindf_list):
    df.to_csv('./dataframes/node_'+str(idx)+'.csv', index=False)

# create initial model
init_model = construct_model(params['architecture'], params['pretrained'], params['device'], params['out_features'])
torch.save(init_model, 'models/agg_model.pt')


@app.route('/getConnection', methods=['GET', 'POST'])
def getConnection():
    print("Connecting Nodes:", end=" ") 
    if len(params['node_names']) < len(params['number_of_samples']):   
        data = request.get_json()
        params['node_names'].append(data['node_number'])
        print(params['node_names'][-1])
        return json.dumps(params)
    else:
        print("Maximum limit reached!")
        return json.dumps({"status":"max_reached"})

@app.route('/getmodel', methods=['GET', 'POST'])
def getModel():
    uploads = 'models'
    filename = 'agg_model.pt'
    return send_from_directory(uploads, filename)

# @app.route('/checkphase', methods=['GET', 'POST'])
# def checkPhase():
#     return json.dumps({'phase': params['phase']})

@app.route('/sendmodel', methods=['GET', 'POST'])
def sendmodel():
    file = request.files['file']
    path = os.path.join('./models', \
        request.files['file'].filename)
    file.save(path)
    result = {"status": "doaggregation"}
    return json.dumps(result)

def aggregation_thread(node_number):
    model_data = []
    node_model = 0
    rem_samples = 0
    for node in params['node_names']:
        if node == str(node_number):
            node_model = torch.load('models/node_'+str(node)+'.pt') \
                .to(params['device'])
            acc, f1 = test(node_model, testloader, params['device'])
            print("node_"+str(node), end=' ')
            print("Test acc:", acc, end=' ')
            print("| F1:", f1)
            node_tuple = (node_model, params['number_of_samples'][int(node)])
            rem_samples = 100 - params['number_of_samples'][int(node)]
            model_data.append(node_tuple)
            break
    agg_model = torch.load('models/agg_model.pt').to(params['device'])
    model_data.append((agg_model, rem_samples))

    if params['aggregator'] == 'fedavg':
        agg_model = fed_avg_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    elif params['aggregator'] == 'comed':
        agg_model = comed_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    
    torch.save(agg_model, 'models/agg_model.pt')
    print("---Aggregation Done---")
    acc, f1 = test(agg_model, testloader, params['device'])
    print("agg_model Test acc:", acc, end=' ')
    print("| F1:", f1)


@app.route('/doaggregation', methods=['GET','POST'])
def doaggregation():
    data = request.get_json()
    number = data['node_number']
    x = threading.Thread(target=aggregation_thread, args=(number))
    x.start()
    return json.dumps({"status": "model sent successfully!"})


@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()