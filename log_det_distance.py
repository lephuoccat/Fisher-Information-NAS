# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:02:35 2021

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
parser.add_argument('--batch-size-train', default=128, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=100, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()
device = 'cuda'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

def variable(t= torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class PreNN(nn.Module):
    def __init__(self):
        super(PreNN, self).__init__()
        self.encoder = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self,X):
        X = self.encoder(X)
        return X
    
class BinaryNN(nn.Module):
    def __init__(self, c):
        super(BinaryNN, self).__init__()
        self.classifier = nn.Linear(10,c)
        
    def forward(self,X):
        X = self.classifier(X)
        return X


# Convolutional Neural Network Architecture for binary/multi-classification
class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(10,128),
        #     nn.ReLU(True),
        #     nn.Linear(128,2))
        self.classifier = nn.Linear(10,2)
        
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
        self.classifier = nn.Linear(10,4)
        
    def forward(self,X):
        X = self.classifier(X)
        return X

# Convolutional Neural Network Architecture for binary/multi-classification
class NN1(nn.Module):
    def __init__(self, c):
        super(NN1, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                # nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                # nn.MaxPool2d(2, stride=2),
                Flatten(),
                nn.Linear(32 * 28 * 28, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 128),
                nn.ReLU(True))
        # c is the number of class label
        self.classifier = nn.Linear(128,c)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X

class NN2(nn.Module):
    def __init__(self, c):
        super(NN2, self).__init__()
        self.encoder = nn.Sequential(
                Flatten(),
                nn.Linear(28 * 28, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True))
        # c is the number of class label
        self.classifier = nn.Linear(128,c)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X

class NN3(nn.Module):
    def __init__(self, c):
        super(NN3, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                Flatten(),
                nn.Linear(64 * 14 * 14, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 128),
                nn.ReLU(True))
        # c is the number of class label
        self.classifier = nn.Linear(128,c)
        
    def forward(self,X):
        X = self.encoder(X)
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

        tensor_target = torch.zeros(args.batch_size_test, 4)
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
dataset = 'MNIST'
# base_task_list = np.array([ [0], 
#                             [6],
#                             [0,1,2,3],
#                             [10] ], dtype=object)

# cifar10 tasks
base_task_list = np.array([ [1,3,8], [3,8,9], [2,6,7], [10] ], dtype=object)

# source and target task ID
source = 2
target = 2

# load the train and test data for indicator task
# if target == 2:
#     print('Loading multi-class dataset...')
#     trainloader, testloader, train_label, test_label = odd_even_dataset(dataset, 10, args)
#     c = 2
# elif target == 3:
#     print('Loading 10-class dataset...')
#     trainloader, testloader, train_label, test_label = full_class_dataset('MNIST', 10, [], args)
#     c = 10
# else:
#     print('Loading binary dataset...')
#     trainloader, testloader, train_label, test_label = indicator_dataset('MNIST', base_task_list[target], 10, [], args)
#     c = 2
            
# load the train and test data for indicator task
if target == 2:
    trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)
elif target == 3:
    # trainloader, testloader, train_label, test_label = CIFAR_dataset(dataset, args)
    trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)
else:
    trainloader, testloader, train_label, test_label = CIFAR_indicator_dataset(dataset, base_task_list[target], args)


# load pre-trained CNN
# prenet = PreNN().cuda()

net = VGG('VGG16').cuda()
# net = ResNet18().cuda()
# net= DenseNet121().cuda()

binary = BinaryNN().cuda()
binary4 = Binary4NN().cuda()


for file_idx in range(10):
    print(file_idx)
    # --------
    # Source
    # --------
    # net_source = NN3(10).cuda()
    net_source = torch.nn.Sequential(net, binary)
    checkpoint = torch.load('./checkpoint/mnist_vgg16/mnist_task'+str(source+0)+'.t'+str(file_idx))
    net_source.load_state_dict(checkpoint['net'])
    
    # calculate fisher matrix
    fisher_matrix_source = diag_fisher_binary(net_source, testloader)
    
    total_source = 0
    for n, p in net_source.named_parameters():
        total_source += np.sum(fisher_matrix_source[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in net_source.named_parameters():
        fisher_matrix_source[n] = fisher_matrix_source[n]/total_source
    
    # --------
    # Target
    # --------
    # net_target = NN3(c).cuda()
    net_target = torch.nn.Sequential(net, binary4)
    checkpoint = torch.load('./checkpoint/cifar10_vgg16/cifar10_task'+str(target+100)+'.t'+str(file_idx))
    net_target.load_state_dict(checkpoint['net'])
    print(checkpoint['acc'])
    
    # calculate fisher matrix
    fisher_matrix_target = diag_fisher(net_target, testloader)
    
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
    
    tensor_target = torch.zeros(4, 10)
    tensor_target[:2,:] = fisher_matrix_source[name[54]]
    fisher_matrix_source[name[54]] = tensor_target.cuda()
    
    tensor_target = torch.zeros(4)
    tensor_target[:2] = fisher_matrix_source[name[55]]
    fisher_matrix_source[name[55]] = tensor_target.cuda()
    
    
    # tensor_target = torch.zeros(10, 10)
    # tensor_target[:4,:] = fisher_matrix_target[name[54]]
    # fisher_matrix_target[name[54]] = tensor_target.cuda()
    
    # tensor_target = torch.zeros(10)
    # tensor_target[:4] = fisher_matrix_target[name[55]]
    # fisher_matrix_target[name[55]] = tensor_target.cuda()
    
    # Frechet distance
    distance = 0
    for n, p in net_target.named_parameters():
        # distance += 0.5 * np.sum(((fisher_matrix_source[n]**0.5 - fisher_matrix_target[n]**0.5)**2).cpu().numpy())
        distance += np.absolute( np.sum( np.log(((fisher_matrix_source[n]+1e-10)/(fisher_matrix_target[n]+1e-10)).cpu().numpy()) ) ) / count_parameters(net_target)
    
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








