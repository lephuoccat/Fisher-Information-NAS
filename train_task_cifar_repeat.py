# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:13:53 2020

@author: catpl
"""
import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from models import *

import numpy as np
import random
from itertools import combinations 
from data_loader import *

device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=50, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=100, type=int, help='number of epochs')
args = parser.parse_args()


def rSubset(arr, r): 
    # return list of all subsets of length r 
    # to deal with duplicate subsets use  
    # set(list(combinations(arr, r))) 
    return list(combinations(arr, r)) 
  
class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    

# Convolutional Neural Network Architecture for binary/multi-classification
class BinaryNN(nn.Module):
    def __init__(self, c):
        super(BinaryNN, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(10,128),
        #     nn.ReLU(True),
        #     nn.Linear(128,2))
        self.classifier = nn.Linear(10,c)
        
    def forward(self,X):
        X = self.classifier(X)
        return X
    
    
# Convolutional Neural Network Architecture for binary/multi-classification
class NN(nn.Module):
    def __init__(self, c):
        super(NN, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                Flatten(),
                nn.Linear(32 * 8 * 8, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 128),
                nn.ReLU(True))
        # c is the number of class label
        self.classifier = nn.Linear(128,c)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X


# Train the CNN
def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = args.num_epoch
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == targets).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} ({:.0f}%) \t\t Accuracy:{:.3f}%'.format(
                    epoch, 100.*batch_idx / len(train_loader), float(correct*100) / float(args.batch_size_train*(batch_idx+1))))


# Test the CNN on test data
best_acc = 0
def evaluate(model, test_loader, label, save_flag, index, i):
    global best_acc
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = test_imgs.to(device)
        test_labels = test_labels.long().to(device)
        
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% \n".format( float(correct * 100) / len(label)))
    
    if (save_flag == True):
        # Save the pretrained network
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': correct,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/cifar10_task'+str(index)+'b.t'+str(i))
    best_acc = correct

# Main Function 
if __name__ == "__main__": 
    for i in range(10):
        
        # task 0, 1, 22, 30
        base_task_list = np.array([ [1,3,8], [3,8,9], [2,6,7], [10] ], dtype=object)
        # base_task_list = np.array([ [1,3,8], [3,8,9] ], dtype=object)
        # base_task_list = np.array([ [2,6,7], [10] ], dtype=object)
        # base_task_list = np.array([ [10] ], dtype=object)
        
        # load CIFAR10 indicator dataset
        for idx in range(len(base_task_list)):
            print(base_task_list[idx])
            
            if idx == 2:
                print('Loading multi-class dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset('CIFAR10', base_task_list[idx], args)
                c = 4
            elif idx == 3:
                print('Loading 10-class dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_dataset('CIFAR10', args)
                c = 10
            else:
                print('Loading binary dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_indicator_dataset('CIFAR10', base_task_list[idx], args)
                c = 2
            
            
            # initialize CNN
            torch.manual_seed(i)
            # net = VGG('VGG16').cuda()
            # net = ResNet18().cuda()
            net = DenseNet121().cuda()
            
            torch.manual_seed(i)
            binary = BinaryNN(c).cuda()
            
            model = torch.nn.Sequential(net, binary)
            print(model)
            
            
            
            # train and evaluate on the indicator task
            fit(model, trainloader)
            evaluate(model, testloader, test_label, True, idx+30, i)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        