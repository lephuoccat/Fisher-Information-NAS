# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:38:46 2020

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
from data_loader import *
from scipy.linalg import sqrtm

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=200, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
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
class CNN(nn.Module):
    def __init__(self, c):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                Flatten(),
                nn.Linear(28 * 28 * 32, 1024),
                nn.ReLU(True))
        # c is the number of class label
        self.classifier = nn.Linear(1024,c)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X


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

        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    
    return precision_matrices





# load dataset
# dataset = 'MNIST'
# dataset = 'fMNIST'
dataset = 'CIFAR10'

# source and target task ID
source = 8 
target = 9
indicator = 0
total_class = 10        # consider all 10 classes of data in MNIST/FashionMNIST/quickdraw
out_dim = 2             # output dimension of CNN

# load the train and test data for indicator task
trainloader, testloader, train_label, test_label = indicator_dataset(dataset, indicator, total_class, class_object, args)


# load pre-trained CNN 1
net_source = CNN(out_dim)
net_source = net_source.to(device)
checkpoint = torch.load('./checkpoint/task'+str(source)+'.t1')
net_source.load_state_dict(checkpoint['net'])


# calculate fisher matrix
fisher_matrix = diag_fisher(net_source, trainloader)

F1 = np.zeros([6,6])
F1[0,0] = np.mean(fisher_matrix['encoder.0.weight'].cpu().numpy())
F1[1,1] = np.mean(fisher_matrix['encoder.0.bias'].cpu().numpy())
F1[2,2] = np.mean(fisher_matrix['encoder.3.weight'].cpu().numpy())
F1[3,3] = np.mean(fisher_matrix['encoder.3.bias'].cpu().numpy())
F1[4,4] = np.mean(fisher_matrix['classifier.weight'].cpu().numpy())
F1[5,5] = np.mean(fisher_matrix['classifier.bias'].cpu().numpy())



# load pre-trained CNN 2
net_target = CNN(out_dim)
net_target = net_target.to(device)
checkpoint = torch.load('./checkpoint/task'+str(target)+'.t1')
net_target.load_state_dict(checkpoint['net'])


# calculate fisher matrix
fisher_matrix = diag_fisher(net_target, trainloader)

F2 = np.zeros([6,6])
F2[0,0] = np.mean(fisher_matrix['encoder.0.weight'].cpu().numpy())
F2[1,1] = np.mean(fisher_matrix['encoder.0.bias'].cpu().numpy())
F2[2,2] = np.mean(fisher_matrix['encoder.3.weight'].cpu().numpy())
F2[3,3] = np.mean(fisher_matrix['encoder.3.bias'].cpu().numpy())
F2[4,4] = np.mean(fisher_matrix['classifier.weight'].cpu().numpy())
F2[5,5] = np.mean(fisher_matrix['classifier.bias'].cpu().numpy())


# Frechet distance
F1_norm = F1 / np.linalg.norm(np.diag(F1))
F2_norm = F2 / np.linalg.norm(np.diag(F2))
distance = 1/2 * np.sum((sqrtm(F1_norm) - sqrtm(F2_norm))**2)
print(distance)




















'''
from torch import autograd

def estimate_fisher(model, data_loader, sample_size, batch_size=32):
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, y in data_loader:
        # x = x.view(batch_size, -1)
        x = x.to(device)
        y = y.to(device)
        A = model(x)
        loglikelihoods.append(
            F.log_softmax(model(x), dim=1)[range(batch_size), y.data]
        )
        # if len(loglikelihoods) >= sample_size // batch_size:
        #     break
    # estimate the fisher information of the parameters.
    loglikelihoods = torch.cat(loglikelihoods).unbind()
    loglikelihood_grads = zip(*[autograd.grad(
        l, model.parameters(),
        retain_graph=(i < len(loglikelihoods))
    ) for i, l in enumerate(loglikelihoods, 1)])
    loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
    fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
    param_names = [
        n.replace('.', '__') for n, p in model.named_parameters()
    ]
    return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}


estimate_fisher(net_target, trainloader, 50, batch_size=32)

'''


