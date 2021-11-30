import torch
import torch.optim as optim
import numpy as np
import random
# from modelloader import getModelArchitecture, loadModel
from model import construct_model
from collections import OrderedDict

def weights_to_array(model):
    '''
        input: pytorch model
        output: array of tensors of model weights
    '''
    model_weights = []
    for (key, param) in model.state_dict().items():
    # for param in model.parameters():
        model_weights.append(param) # check without .data
    return model_weights

def fed_avg_aggregator(model_data, architecture, pretrained, device, out_features):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: fed_avg aggregated model
    '''
    total_no_samples = 0
    
    # creates an array of weights and sample counts
    # shape -> no_models*no_layers*dim_of_layer
    node_weights = []
    node_samples = []
    for model,no_samples in model_data:
        node_weights.append(weights_to_array(model))
        node_samples.append(no_samples)
    # calculates the total number of samples
        total_no_samples += no_samples
    
    aggregated_weights = []
    for layer_idx in range(len(node_weights[0])):
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(device)
        for node_idx in range(len(node_weights)):
            fraction = (node_samples[node_idx]/total_no_samples)
            temp+= fraction*node_weights[node_idx][layer_idx]
        aggregated_weights.append(temp)
    agg_model = construct_model(architecture, pretrained, device, out_features)
    
    agg_state = OrderedDict()
    for idx, key in enumerate(agg_model.state_dict().keys()):
        agg_state[key] = aggregated_weights[idx]
    agg_model.load_state_dict(agg_state)
#     agg_model = loadModel(args['aggregated_model_location']+'agg_model.pt').to(args['device'])
#     for idx, (key, param) in enumerate(agg_model.state_dict().items()):
#         agg_model.state_dict()[key] = aggregated_weights[idx]
#         import pdb; pdb.set_trace()
    return agg_model

#COMED aggregator
def comed_aggregator(model_data, architecture, pretrained, device, out_features):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: COMED aggregated model
    '''
    total_no_samples = 0
    
    # creates an array of weights and sample counts
    # shape -> no_models*no_layers*dim_of_layer
    node_weights = []
    node_samples = []

    for model,no_samples in model_data:
        node_weights.append(weights_to_array(model))
        node_samples.append(no_samples)
        total_no_samples += no_samples
     
    aggregated_weights = []
    for layer_idx in range(len(node_weights[0])):
        layer_shape = node_weights[0][layer_idx].shape
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(device)
        for node_idx in range(len(node_weights)):
            if(node_idx == 0):
                temp = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
            else:
                layer_flattened = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
                temp = torch.cat((temp, layer_flattened),1)
        temp = temp.detach().cpu().numpy()
        temp = np.median(temp,1)
        temp = torch.from_numpy(temp)
        temp = torch.reshape(temp, layer_shape)
        aggregated_weights.append(temp)

    agg_model = construct_model(architecture, pretrained, device, out_features)

    agg_state = OrderedDict()
    for idx, key in enumerate(agg_model.state_dict().keys()):
        agg_state[key] = aggregated_weights[idx]
    agg_model.load_state_dict(agg_state)
    
    return agg_model
