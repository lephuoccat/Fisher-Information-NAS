# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:18:42 2020

@author: catpl
"""

import os
import argparse

import numpy as np
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

from models import *
from data_loader import *
from scipy.linalg import sqrtm
# import matplotlib.pyplot as plt

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=10, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()
device = 'cuda'


class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

# Convolutional Neural Network Architecture for binary/multi-classification
class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(10,128),
        #     nn.ReLU(True),
        #     nn.Linear(128,2))
        self.classifier = nn.Linear(10,4)
        
    def forward(self,X):
        X = self.classifier(X)
        return X

class Binary4NN(nn.Module):
    def __init__(self):
        super(Binary4NN, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(10,128),
        #     nn.ReLU(True),
        #     nn.Linear(128,2))
        self.classifier = nn.Linear(10,10)
        
    def forward(self,X):
        X = self.classifier(X)
        return X


def diag_fisher_binary(model, data):
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)

    model.eval()
    error = nn.CrossEntropyLoss()
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        
        tensor_target = torch.zeros(10, 21)
        tensor_target[:,:2] = output
        output = tensor_target.cuda()
        
        # print(output.shape)

        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    
    return precision_matrices


def diag_fisher(model, data):
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)

    model.eval()
    error = nn.CrossEntropyLoss()
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        # print(output.shape)

        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    
    return precision_matrices



# load dataset
dataset = 'CIFAR10'
# base_task_list = np.array([ [1,3,8], [3,8,9], [3,4,8], [0,3,9], [0,5,9], [0,2,9], [5,6,7], [2,6,7], [0], [0], 
#                             [1], [9], [7], [0], [0], [0], [0], [0], [0], [0],
#                             [1,3,8], [0,3,9], [2,6,7]])

base_task_list = np.array([ [1,3,8], [3,8,9], [2,6,7], [10] ], dtype=object)

# base_task_list = np.array([ [8, 58, 90, 13, 48, 81, 69, 41, 89, 85], 
#                             [40, 39, 22, 87, 86, 20, 25, 94, 84, 5], 
#                             [8, 58, 90, 13, 48, 81, 69, 41, 89, 85], 
#                             [8, 58, 90, 13, 48, 81, 69, 41, 89, 85, 40, 39, 22, 87, 86, 20, 25, 94, 84, 5] ])
# source and target task ID
source = 2
target = 2

# load the train and test data for indicator task
if target == 2:
    trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)
elif target == 3:
    # trainloader, testloader, train_label, test_label = CIFAR_dataset(dataset, args)
    trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)
else:
    trainloader, testloader, train_label, test_label = CIFAR_indicator_dataset(dataset, base_task_list[target], args)

# load pre-trained CNN
# net = VGG('VGG16').cuda()
# net = ResNet18().cuda()
net= DenseNet121().cuda()
binary = BinaryNN().cuda()
binary4 = Binary4NN().cuda()


# file_idx = 0
for file_idx in range(10):
    print(file_idx)
    # --------
    # Source
    # --------
    net_source = torch.nn.Sequential(net, binary)
    net_source = net_source.to(device)
    # checkpoint = torch.load('./checkpoint/cifar10_task'+str(source)+'.t1')
    # checkpoint = torch.load('./checkpoint/cifar100_densenet121/cifar100_task'+str(source+20)+'.t'+str(file_idx))
    checkpoint = torch.load('./checkpoint/cifar10_task'+str(source+30)+'b.t'+str(file_idx))
    net_source.load_state_dict(checkpoint['net'])
    print(checkpoint['acc'])
    
    # calculate fisher matrix
    fisher_matrix_source = diag_fisher(net_source, testloader)
    
    total_source = 0
    for n, p in net_source.named_parameters():
        total_source += np.sum(fisher_matrix_source[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in net_source.named_parameters():
        fisher_matrix_source[n] = fisher_matrix_source[n]/total_source
    
    # --------
    # Target
    # --------
    net_target = torch.nn.Sequential(net, binary)
    net_target = net_target.to(device)
    # checkpoint = torch.load('./checkpoint/cifar10_task'+str(target)+'.t1')
    checkpoint = torch.load('./checkpoint/cifar100_densenet121/cifar100_task'+str(target+20)+'.t'+str(file_idx))
    net_target.load_state_dict(checkpoint['net'])
    
    # calculate fisher matrix
    fisher_matrix_target = diag_fisher_binary(net_target, testloader)
    
    total_target = 0
    for n, p in net_target.named_parameters():
        total_target += np.sum(fisher_matrix_target[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in net_target.named_parameters():
        fisher_matrix_target[n] = fisher_matrix_target[n]/total_target
    
    
    # obtain name of layers
    name = {}
    idx = 0
    for n, p in net_source.named_parameters():
        name[idx] = n
        idx += 1
        
    # change dimension of layer 54 and 55
    # vgg16: 54 and 55 
    # resnet18: 62 and 63
    # densenet121:362 and 363
    
    # tensor_target = torch.zeros(21, 10)
    # tensor_target[:11,:] = fisher_matrix_source[name[362]]
    # fisher_matrix_source[name[362]] = tensor_target.cuda()
    
    # tensor_target = torch.zeros(21)
    # tensor_target[:11] = fisher_matrix_source[name[363]]
    # fisher_matrix_source[name[363]] = tensor_target.cuda()
    
    
    tensor_target = torch.zeros(21, 10)
    tensor_target[:2,:] = fisher_matrix_target[name[362]]
    fisher_matrix_target[name[362]] = tensor_target.cuda()
    
    tensor_target = torch.zeros(21)
    tensor_target[:2] = fisher_matrix_target[name[363]]
    fisher_matrix_target[name[363]] = tensor_target.cuda()
    
    # Frechet distance
    distance = 0
    for n, p in net_target.named_parameters():
        distance += 0.5 * np.sum(((fisher_matrix_source[n]**0.5 - fisher_matrix_target[n]**0.5)**2).cpu().numpy())
    
    print(distance)

    

    
    

'''

# --------------------------
# layer-by-layer
# --------------------------
# Source
net_source = torch.nn.Sequential(net, binary)
net_source = net_source.to(device)
checkpoint = torch.load('./checkpoint/cifar10_task'+str(source)+'.t3')
net_source.load_state_dict(checkpoint['net'])
fisher_matrix_source = diag_fisher_binary(net_source, testloader)

# obtain name of layers
name = {}
idx = 0
for n, p in net_source.named_parameters():
    name[idx] = n
    idx += 1

# normalize each layer
for n, p in net_source.named_parameters():
    total_source = np.sum(fisher_matrix_source[n].cpu().numpy())
    fisher_matrix_source[n] = fisher_matrix_source[n]/total_source

# change dimension of layer 54 and 55
# tensor_target = torch.zeros(10, 10)
# tensor_target[:2,:] = fisher_matrix_source[name[len(name)-2]]
# fisher_matrix_source[name[len(name)-2]] = tensor_target.cuda()

# tensor_target = torch.zeros(10)
# tensor_target[:2] = fisher_matrix_source[name[len(name)-1]]
# fisher_matrix_source[name[len(name)-1]] = tensor_target.cuda()



# Target
net_target = torch.nn.Sequential(net, binary4)
net_target = net_target.to(device)
checkpoint = torch.load('./checkpoint/cifar10_task'+str(target)+'.t3')
net_target.load_state_dict(checkpoint['net'])
fisher_matrix_target = diag_fisher(net_target, testloader)
# normalize each layer
for n, p in net_target.named_parameters():
    total_target = np.sum(fisher_matrix_target[n].cpu().numpy())
    fisher_matrix_target[n] = fisher_matrix_target[n]/total_target
    
# change dimension of layer 54 and 55
# tensor_target = torch.zeros(4, 10)
# tensor_target[:2,:] = fisher_matrix_target[name[len(name)-2]]
# fisher_matrix_target[name[len(name)-2]] = tensor_target.cuda()

# tensor_target = torch.zeros(4)
# tensor_target[:2] = fisher_matrix_target[name[len(name)-1]]
# fisher_matrix_target[name[len(name)-1]] = tensor_target.cuda()    
    

# Frechet distance layer-by-layer
distance = []
for n, p in net_target.named_parameters():
    distance_layer = 0.5 * np.sum(((fisher_matrix_source[n]**0.5 - fisher_matrix_target[n]**0.5)**2).cpu().numpy())
    distance.append(distance_layer)
distance = np.array(distance)


# distance_weight = distance[np.arange(0,len(distance),2)]
# distance_conv = distance_weight[np.arange(0,len(distance_weight)-2,2)]
# distance_final = np.concatenate((distance_conv, distance_weight[-2:]), axis=0)
# plt.plot(distance_final)

distance_weight = distance[np.arange(0,len(distance)-4,3)]
distance_conv = distance[np.arange(len(distance)-4, len(distance),2)]
distance_final = np.concatenate((distance_weight, distance_conv), axis=0)
plt.plot(distance_final)






# plot multiple lines
# x = np.arange(0,15)
# plt.plot(x,t0_t1,label='d[t0,t1]')
# plt.plot(x,t0_t2,label='d[t0,t2]')
# plt.plot(x,t0_t6,label='d[t0,t6]')
# plt.plot(x,t0_t7,label='d[t0,t7]')

# plt.legend(loc="upper left")
# plt.ylim(0, 1)
# plt.show()



# -----------------------------------------------
# padding for different task of classification
# -----------------------------------------------
tensor_target = torch.zeros(4, 10)
tensor_target[:2,:] = fisher_matrix_source[name[54]]

'''








