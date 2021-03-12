# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:50:01 2020

@author: catpl
"""

import os
import argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class feature_Dataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        

# load data for num-indicator
def CIFAR10_indicator_dataset(dataset, num, args):
    print("Loading full CIFAR10 dataset...")
    data_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # -------------------------
    # train batch
    # -------------------------
    train_label = np.array(data_train.targets)
    train_label[train_label == num[0]] = 10
    train_label[train_label == num[1]] = 10
    train_label[train_label == num[2]] = 10
    train_label[train_label != 10] = 0
    train_label[train_label == 10] = 1
    
    # select only 15,000 datapoint with label 0
    idx_1 = (train_label == 1)
    idx_0 = (train_label == 0)
    sum_idx_0 = 0
    total = sum(idx_1)
    
    for i in range(len(idx_0)):
        sum_idx_0 += idx_0[i]
        
        if sum_idx_0 == total:
            idx_0[i+1:] = False
            break
    # all index 0 and 1
    idx = idx_0 + idx_1
    print(sum(idx))
    # update data_train data and label
    train_label = train_label[idx]
    data_train.targets = train_label.tolist()
    data_train.data = data_train.data[idx]
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=False)
    
    # -------------------------
    # test batch
    # -------------------------
    test_label = np.array(data_test.targets)
    test_label[test_label == num[0]] = 10
    test_label[test_label == num[1]] = 10
    test_label[test_label == num[2]] = 10
    test_label[test_label != 10] = 0
    test_label[test_label == 10] = 1
    
    # select only 3,000 datapoint with label 0
    idx_1 = (test_label == 1)
    idx_0 = (test_label == 0)
    sum_idx_0 = 0
    total = sum(idx_1)
    
    for i in range(len(idx_0)):
        sum_idx_0 += idx_0[i]
        
        if sum_idx_0 == total:
            idx_0[i+1:] = False
            break
    # all index 0 and 1
    idx = idx_0 + idx_1
    print(sum(idx))
    # update data_train data and label
    test_label = test_label[idx]
    data_test.targets = test_label.tolist()
    data_test.data = data_test.data[idx]
    testloader = DataLoader(data_test, batch_size=args.batch_size_train, shuffle=False)
    
    return trainloader, testloader, train_label, test_label





parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=200, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()

trainloader, testloader, train_label, test_label = indicator_dataset('CIFAR10', [1,2,3], args)



