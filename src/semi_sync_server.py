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
import csv
from flask import Flask, request, send_from_directory, send_file
import requests
import json
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
import socket
print("ip address: ", socket.gethostbyname(socket.gethostname()))


for i in os.listdir('models'):
    os.remove('models/'+i)

for i in os.listdir('client_models'):
    os.remove('client_models/'+i)

params = {
    'node_names': [],
    'threshold': 0.5,
    'model_queue': [],
    'number_of_samples' : [10,20, 30, 40],
    'device' : 'cpu',
    'architecture' : 'simplenet',
    'batch_size' : 4,
    'number_of_iterations' : 50,
    'number_of_epochs' : 1,
    'learning_rate' : 0.01,
    'pretrained' : True,
    'aggregator' : 'comed', #fedavg or comed
    'out_features' : 10,
    'count_done': 0, 
    'phase': 0 #init, aggregating, training
}

start_time = time()
file_name_str = 'semi_'+params['architecture']+'_'+params['aggregator']+'_'+str(len(params['number_of_samples']))
f = open('../src/results/'+file_name_str+'.csv', 'w')
writer = csv.writer(f)
writer.writerow(['time', 'acc', 'f1'])

#create dfs
traindf_list = generate_train_data(params['number_of_samples'])
test_df = generate_test_data()
test_df.to_csv('./dataframes/test.csv', index=False)
testloader = get_test_dataloader(test_df, params['device'])

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

@app.route('/checkphase', methods=['GET', 'POST'])
def checkPhase():
    number = request.get_json()['number']
    if number in params['model_queue']:
        return json.dumps({'status': 'wait'})
    else:
        return json.dumps({'status': 'process'})

@app.route('/sendmodel', methods=['GET', 'POST'])
def sendmodel():
    file = request.files['file']
    path = os.path.join('./models', \
        request.files['file'].filename)
    file.save(path)
    
    params['model_queue'].append(request.files['file'].filename.split('.')[0].split('_')[1])
    if (len(params['model_queue']) > len(params['number_of_samples']))*params['threshold'] or \
        (len(params['model_queue']) == len(params['number_of_samples'])*params['threshold']):
        result = {"status": "doaggregation"}
    else:
        result = {"status": "model sent successfully!"}
    return json.dumps(result)

def aggregation_thread():
    model_data = []
    node_model = 0
    for node in params['node_names']:
        if str(node) in params['model_queue']:
            node_model = torch.load('models/node_'+str(node)+'.pt') \
                .to(params['device'])
            acc, f1 = test(node_model, testloader, params['device'])
            print("node_"+str(node), end=' ')
            print("Test acc:", acc, end=' ')
            print("| F1:", f1)
            # test(serverargs, node_model, test_loader, logger=logger)
            node_tuple = (node_model, params['number_of_samples'][int(node)])
            model_data.append(node_tuple)
    print("collected nodes: ", [i for (model, i) in model_data])
    if params['aggregator'] == 'fedavg':
        agg_model = fed_avg_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    elif params['aggregator'] == 'comed':
        agg_model = comed_aggregator(model_data, params['architecture'], params['pretrained'], \
            params['device'], params['out_features'])
    
    torch.save(agg_model, 'models/agg_model.pt')
    print("---Aggregation Done---")
    params['model_queue'] = []
    acc, f1 = test(agg_model, testloader, params['device'])
    print("agg_model Test acc:", acc, end=' ')
    print("| F1:", f1)
    abs_time = time() - start_time
    f = open('../src/results/'+file_name_str+'.csv', 'w')
    writer = csv.writer(f)
    writer.writerow([abs_time, acc, f1])
    f.close()
    if abs_time > 5000:
        exit()
    


@app.route('/doaggregation', methods=['GET','POST'])
def doaggregation():
    x = threading.Thread(target=aggregation_thread, args=())
    x.start()
    return json.dumps({"status": "model sent successfully!"})


@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(host=socket.gethostbyname(socket.gethostname()),port=5000)