# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:12:24 2020

@author: catpl
"""

import os
import argparse

import numpy as np
import random as rd

import torch
import torchvision
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from operation import *
from data_loader import *

device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=20, type=int, help='number of epochs')
args = parser.parse_args()

# Dataset
if not os.path.exists('./feature_data'):
    os.mkdir('./feature_data')

class Cell(nn.Module):
    def __init__(self, dim, num_node=3, reduction=0, channel=1):
        super(Cell, self).__init__()
        stride = 2 if reduction==1  else 1
        
        self.op_0_1 = OPS[draw_operation(9)](channel, stride, affine=False)
        self.op_0_2 = OPS[draw_operation(9)](channel, stride, affine=False)
        self.op_1_2 = OPS[draw_operation(9)](channel, stride, affine=False)
        
        self.classifier = nn.Linear(2 * 28 * 28, dim)

    def forward(self, X):
        s01 = self.op_0_1(X)
        s02 = self.op_0_2(X)
        s12 = self.op_1_2(s01)

        # out = s02 + s12
        out = torch.cat([s02, s12], dim=1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


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
                print('Epoch : {} ({:.0f}%)\t Accuracy:{:.3f}%'.format(
                    epoch, 100.*batch_idx / len(train_loader), float(correct*100) / float(args.batch_size_train*(batch_idx+1))))


# Test the CNN on test data
best_acc = 0
def evaluate(model, test_loader, label, save_flag, index):
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
        torch.save(state, './checkpoint/indicator'+str(index)+'.t1')
    best_acc = correct
    
# ------------
# main file
# ------------
# task id
taskID = 100

# dataset = 'MNIST'
#dataset = 'fMNIST'
dataset = 'quickdraw'

# quickdraw! class object
class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']

indicator_idx = 1
total_class = 10        # consider all 10 classes of data in MNIST/FashionMNIST/quickdraw

# output dimension of CNN
# dim = total_class for full-class classification, dim = 2 for binary classification
out_dim = 2

# load the train and test data for full-class task
# trainloader, testloader, train_label, test_label = full_class_dataset(dataset, total_class, class_object, args)

# load the train and test data for indicator task
trainloader, testloader, train_label, test_label = indicator_dataset(dataset, indicator_idx, total_class, class_object, args)

# load the train and test data for odd vs even task
# trainloader, testloader, train_label, test_label = odd_even_dataset(dataset, total_class, args)


# CNN model
torch.manual_seed(0)
cnn = Cell(out_dim).cuda()
print(cnn)
fit(cnn, trainloader)
evaluate(cnn, testloader, test_label, False, taskID)





