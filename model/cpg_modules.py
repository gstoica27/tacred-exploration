import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import itertools
import numbers
import numpy as np
import os
from functools import reduce
from operator import mul


# Contextual Parameter Generator Class for ConvE and other methods
# network_structure: dimensions of input to all hidden layers of network
# - input is assumed to be first element of network_structure
# - last element is assumed to be last hidden layer of network
# output_shape: dimensions of the desired paired network parameters
# dropout: probability for dropout
# use_batch_norm: whether to use batch_norm
# batch_norm_momentum: momentum for batchnorm
# use_bias: whether CPG network should have bias
class ContextualParameterGenerator(nn.Module):
    def __init__(self, network_structure, output_shape, dropout, use_batch_norm=False,
                 batch_norm_momentum=0.99, use_bias=False):
        super(ContextualParameterGenerator, self).__init__()
        self.network_structure = network_structure
        self.output_shape = output_shape
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        # print('use bias: {}'.format(self.use_bias))
        self.flattened_output = reduce(mul, output_shape, 1)
        #print('input shape: {}'.format(network_structure[0]))
        self.projections = nn.ModuleList([])
        layer_input = network_structure[0]
        print('network structure: {}'.format(network_structure))
        for layer_output in self.network_structure[1:]:
            print('inside loop!')
            self.projections.append(nn.Linear(layer_input, layer_output, bias=self.use_bias))
            #for name, param in self.projections[-1].named_parameters():
             #   print('{} size: {}'.format(name, param.size()))
            if use_batch_norm:
                self.projections.append(nn.BatchNorm1d(num_features=layer_output,
                                                       momentum=batch_norm_momentum))
                #for name, param in self.projections[-1].named_parameters():
                 #   print('Batch Norm | {} size: {}'.format(name, param.size()))
            self.projections.append(nn.ReLU())
            self.projections.append(nn.Dropout(p=self.dropout))
            layer_input = layer_output
        print('creating output layer')
        self.projections.append(nn.Linear(layer_input, self.flattened_output, bias=self.use_bias))
        #print('Printing Network Architecture: ')
        #for name, param in self.projections.named_parameters():
        #    print('| {} size: {}'.format(name, param.size()))
        self.network = nn.Sequential(*self.projections)

    def forward(self, query_emb):
        #print('the device of the CPG network is: {}'.format(self.network.device))
        # print('query embedding device: {}'.format(query_emb.device))
        flat_params = self.network(query_emb)
        params = flat_params.view([-1] + self.output_shape)
        # print('CPG shape: {}'.format(params.shape))
        return params

