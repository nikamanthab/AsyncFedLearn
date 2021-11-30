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

params = {
    'node_names': [],
    'model_list' : [],
    'number_of_samples' : [100],
    'device' : 'cpu',
    'architecture' : 'simplenet',
    'batch_size' : 2,
    'number_of_iterations' : 50,
    'number_of_epochs' : 1,
    'learning_rate' : 0.01,
    'pretrained' : True,
    'aggregator' : 'fedavg', #fedavg or comed
    'out_features' : 10,
    'count_done': 0, 
    'phase': 0 #init, aggregating, training
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
params['model_list'] = ['NA']*len(params['number_of_samples'])


@app.route('/getConnection', methods=['GET', 'POST'])
def getConnection():
    print(params['phase'])
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
    print(params['phase'])
    uploads = 'models'
    filename = 'agg_model.pt'
    return send_from_directory(uploads, filename)

@app.route('/checkphase', methods=['GET', 'POST'])
def checkPhase():
    return json.dumps({'phase': params['phase']})
    # print(params['phase'])
    # if params['phase'] == 'init' and len(params['node_names']) == len(params['number_of_samples']):
    #     params['phase'] = 'training'
    #     return json.dumps({"phase": "done"})
    # if params['phase'] == 'training':
    #     return json.dumps({"phase": "training"})
    # else:
    #     return json.dumps({"phase": "init"})


@app.route('/sendmodel', methods=['GET', 'POST'])
def sendmodel():
    print(params['phase'])
    file = request.files['file']
    path = os.path.join('./models', \
        request.files['file'].filename)
    file.save(path)
    
    node_model = torch.load(path)
    acc, f1 = test(node_model, testloader, params['device'])
    print("Test acc:", acc, end=' ')
    print("| F1:", f1)
    
    params['count_done']+=1
    if params['count_done'] == len(params['number_of_samples']):
        result = {"status": "doaggregation"}
    else:
        result = {"status": "model sent successfully!"}
    return json.dumps(result)

def aggregation_thread():
    model_data = []
    node_model = 0
    for node in params['node_names']:
        node_model = torch.load('models/node_'+str(node)+'.pt') \
            .to(params['device'])
        acc, f1 = test(node_model, testloader, params['device'])
        print("Test acc:", acc, end=' ')
        print("| F1:", f1)
        # test(serverargs, node_model, test_loader, logger=logger)
        node_tuple = (node_model, params['number_of_samples'][int(node)])
        model_data.append(node_tuple)

    if params['aggregator'] == 'fedavg':
        agg_model = fed_avg_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    elif params['aggregator'] == 'comed':
        agg_model = comed_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    
    torch.save(agg_model, 'models/agg_model.pt')
    print("---Aggregation Done---")
    params['phase'] += 1
    params['count_done'] = 0
    acc, f1 = test(agg_model, testloader, params['device'])
    print("Test acc:", acc, end=' ')
    print("| F1:", f1)


@app.route('/doaggregation', methods=['GET','POST'])
def doaggregation():
    print(params['phase'])
    x = threading.Thread(target=aggregation_thread, args=())
    x.start()
    return json.dumps({"status": "model sent successfully!"})


@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()