# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:57:38 2020

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

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

# quickdraw! class object
class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']

transform = transforms.Compose([
    transforms.Pad(padding=2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

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
        

# load full limit-class
def full_class_dataset(dataset, limit, class_object, args):
    if (dataset == 'MNIST'):
        print("Loading full MNIST dataset...")
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif (dataset == 'fMNIST'):
        print("Loading full Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading full QuickDraw! dataset...")
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(class_object)):
            # load npy file and concatenate data
            ob = np.load('./data/quickdraw/full_numpy_bitmap_'+ class_object[i] +'.npy')
            # choose train size and test size
            train = ob[0:5000,]
            test = ob[5000:6000,]
            train_label = np.concatenate((train_label, i * np.ones(train.shape[0])), axis=0)
            test_label = np.concatenate((test_label, i * np.ones(test.shape[0])), axis=0)
            
            if i == 0:
                train_data = train
                test_data = test
            else:
                train_data = np.concatenate((train_data, train), axis=0)
                test_data = np.concatenate((test_data, test), axis=0)
        
        # generate dataloader
        trainset = feature_Dataset(train_data, train_label, transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
        
        testset = feature_Dataset(test_data, test_label, transform)
        testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
                    
        
    if (dataset == 'MNIST' or dataset == 'fMNIST'):
        # train batch
        idx = (data_train.targets < limit)
        data_train.targets = data_train.targets[idx]
        
        print("Changing label...")
        data_train.targets[data_train.targets == 1] = 10
        data_train.targets[data_train.targets == 6] = 1
        data_train.targets[data_train.targets == 10] = 6
        
        
        data_train.data = data_train.data[idx]
        train_label = data_train.targets.cpu().detach().numpy()
        trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
        # test batch
        idx = (data_test.targets < limit)
        data_test.targets = data_test.targets[idx]
        
        print("Changing label...")
        data_test.targets[data_test.targets == 1] = 10
        data_test.targets[data_test.targets == 6] = 1
        data_test.targets[data_test.targets == 10] = 6
        
        data_test.data = data_test.data[idx]
        test_label = data_test.targets.cpu().detach().numpy()
        testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


# load data for num-indicator
def indicator_dataset(dataset, num, limit, class_object, args):
    if (dataset == 'MNIST'):
        print("Loading {}-indicator for MNIST dataset...".format(num))
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif (dataset == 'fMNIST'):
        print("Loading full Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading full QuickDraw! dataset...")
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(class_object)):
            # load npy file and concatenate data
            ob = np.load('./data/quickdraw/full_numpy_bitmap_'+ class_object[i] +'.npy')
            # choose train size and test size
            train = ob[0:5000,]
            test = ob[5000:6000,]
            train_label = np.concatenate((train_label, i * np.ones(train.shape[0])), axis=0)
            test_label = np.concatenate((test_label, i * np.ones(test.shape[0])), axis=0)
            
            if i == 0:
                train_data = train
                test_data = test
            else:
                train_data = np.concatenate((train_data, train), axis=0)
                test_data = np.concatenate((test_data, test), axis=0)
        
        train_label[train_label != num] = -1
        train_label[train_label == num] = 1
        train_label[train_label == -1] = 0
        
        test_label[test_label != num] = -1
        test_label[test_label == num] = 1
        test_label[test_label == -1] = 0
        
        # generate dataloader
        trainset = feature_Dataset(train_data, train_label.astype(int), transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
        
        testset = feature_Dataset(test_data, test_label.astype(int), transform)
        testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
    
    if (dataset == 'MNIST' or dataset == 'fMNIST'):
        # train batch
        idx = (data_train.targets < limit)
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        
        print("Changing label...")
        data_train.targets[data_train.targets == 1] = 10
        data_train.targets[data_train.targets == 6] = 1
        data_train.targets[data_train.targets == 10] = 6
        
        for i in num:
            data_train.targets[data_train.targets == i] = 10
        
        data_train.targets[data_train.targets != 10] = 0
        data_train.targets[data_train.targets == 10] = 1
        
        idx_0 = (data_train.targets  == 0)
        idx_1 = (data_train.targets == 1)
        sum_idx_0 = 0
        total = sum(idx_1)
        
        for i in range(len(idx_0)):
            sum_idx_0 += idx_0[i]
            
            if sum_idx_0 == total:
                idx_0[i+1:] = False
                break
            
        idx = idx_0 + idx_1
        print(sum(idx))
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        
        train_label = data_train.targets.cpu().detach().numpy()
        trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
        
        # test batch
        idx = (data_test.targets < limit)
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        
        print("Changing label...")
        data_test.targets[data_test.targets == 1] = 10
        data_test.targets[data_test.targets == 6] = 1
        data_test.targets[data_test.targets == 10] = 6
        
        for i in num:
            data_test.targets[data_test.targets == i] = 10
            
        data_test.targets[data_test.targets != 10] = 0
        data_test.targets[data_test.targets == 10] = 1
        
        idx_0 = (data_test.targets  == 0)
        idx_1 = (data_test.targets == 1)
        sum_idx_0 = 0
        print(sum(idx_1))
        # total = sum(idx_1)
        total = 1042
        
        for i in range(len(idx_0)):
            sum_idx_0 += idx_0[i]
            
            if sum_idx_0 == total:
                idx_0[i+1:] = False
                break
            
        idx = idx_0 + idx_1
        print(sum(idx))
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        
        test_label = data_test.targets.cpu().detach().numpy()
        testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


# odd vs even tasks
def odd_even_dataset(dataset, limit, args):
    if (dataset == 'MNIST'):
        print("Loading odd vs even MNIST dataset...")
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading odd vs even Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # train batch
    idx = (data_train.targets < limit)
    data_train.targets = data_train.targets[idx]
    data_train.data = data_train.data[idx]
    data_train.targets[(data_train.targets % 2) == 0] = 10
    data_train.targets[(data_train.targets % 2) != 0] = 0
    data_train.targets[data_train.targets == 10] = 1   
    train_label = data_train.targets.cpu().detach().numpy()
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    # test batch
    idx = (data_test.targets < limit)
    data_test.targets = data_test.targets[idx]
    data_test.data = data_test.data[idx]
    data_test.targets[(data_test.targets % 2) == 0] = 10
    data_test.targets[(data_test.targets % 2) != 0] = 0
    data_test.targets[data_test.targets == 10] = 1
    test_label = data_test.targets.cpu().detach().numpy()
    testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label




# load data for num-indicator
def multi_indicator_dataset(dataset, num, limit, class_object, args):
    if (dataset == 'MNIST'):
        print("Loading {}-multi-indicator for MNIST dataset...".format(num))
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif (dataset == 'fMNIST'):
        print("Loading full Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading full QuickDraw! dataset...")
        
    
    if (dataset == 'MNIST' or dataset == 'fMNIST'):
        # train batch
        idx = (data_train.targets < limit)
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        
        idx = 1
        for i in num:
            data_train.targets[data_train.targets == i] = 10 + idx
            print('adding...')
            idx += 1
        
        data_train.targets[data_train.targets < 10] = 0
        data_train.targets[data_train.targets > 10] -= 10
        
        idx_0 = (data_train.targets  == 0)
        idx_1 = (data_train.targets != 0)
        sum_idx_0 = 0
        total = sum(idx_1)//len(num)
        
        for i in range(len(idx_0)):
            sum_idx_0 += idx_0[i]
            
            if sum_idx_0 == total:
                idx_0[i+1:] = False
                break
            
        idx = idx_0 + idx_1
        print(sum(idx))
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        
        train_label = data_train.targets.cpu().detach().numpy()
        trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
        
        # test batch
        idx = (data_test.targets < limit)
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        
        idx = 1
        for i in num:
            data_test.targets[data_test.targets == i] = 10 + idx
            idx += 1
        
        data_test.targets[data_test.targets < 10] = 0
        data_test.targets[data_test.targets > 10] -= 10
        
        idx_0 = (data_test.targets  == 0)
        idx_1 = (data_test.targets != 0)
        sum_idx_0 = 0
        total = sum(idx_1)//len(num)
        print(sum(idx_1))
        # total = 843
        
        for i in range(len(idx_0)):
            sum_idx_0 += idx_0[i]
            
            if sum_idx_0 == total:
                idx_0[i+1:] = False
                break
            
        idx = idx_0 + idx_1
        print(sum(idx))
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        
        test_label = data_test.targets.cpu().detach().numpy()
        testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label

#---------------------------------------------------------------------------------------------------
# CIFAR dataset
#---------------------------------------------------------------------------------------------------

# transform_cifar = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR dataset
def CIFAR_dataset(dataset, args):
    if (dataset == 'CIFAR10'):
        print("Loading full CIFAR10 dataset...")
        data_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        print("Loading full CIFAR100 dataset...")
        data_train = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
    # -------------------------
    # train batch
    # -------------------------
    train_label = np.array(data_train.targets)
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    
    # -------------------------
    # test batch
    # -------------------------
    test_label = np.array(data_test.targets)
    testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


def CIFAR_indicator_dataset(dataset, num, args):
    if (dataset == 'CIFAR10'):
        print("Loading indicator CIFAR10 dataset...")
        data_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        print("Loading indicator CIFAR100 dataset...")
        data_train = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
    # -------------------------
    # train batch
    # -------------------------
    train_label = np.array(data_train.targets)
    for i in num:
        train_label[train_label == i] = 100
    
    train_label[train_label != 100] = 0
    train_label[train_label == 100] = 1
    
    # select only X datapoint with label 0 to obtain balanced dataset
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
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    
    # -------------------------
    # test batch
    # -------------------------
    test_label = np.array(data_test.targets)
    for i in num:
        test_label[test_label == i] = 100
    
    test_label[test_label != 100] = 0
    test_label[test_label == 100] = 1
    
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
    testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label



def CIFAR_multi_indicator_dataset(dataset, num, args):
    if (dataset == 'CIFAR10'):
        print("Loading multi-class CIFAR10 dataset...")
        data_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        print("Loading multi-class CIFAR100 dataset...")
        data_train = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        data_test = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    # -------------------------
    # train batch
    # -------------------------
    train_label = np.array(data_train.targets)
    idx = 1
    for i in num:
        train_label[train_label == i] = 100 + idx
        idx += 1
    
    train_label[train_label < 100] = 0
    train_label[train_label > 100] -= 100
        
    # select only 15,000 datapoint with label 0
    idx_1 = (train_label != 0)
    idx_0 = (train_label == 0)
    sum_idx_0 = 0
    total = sum(idx_1)/len(num)
    
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
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    
    # -------------------------
    # test batch
    # -------------------------
    test_label = np.array(data_test.targets)
    idx = 1
    for i in num:
        test_label[test_label == i] = 100 + idx
        idx += 1
    
    test_label[test_label < 100] = 0
    test_label[test_label > 100] -= 100
    
    # select only 3,000 datapoint with label 0
    idx_1 = (test_label != 0)
    idx_0 = (test_label == 0)
    sum_idx_0 = 0
    total = sum(idx_1)/len(num)
    
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
    testloader = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


