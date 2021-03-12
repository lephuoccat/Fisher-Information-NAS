# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:12:08 2021

@author: catpl
"""
import os
import argparse

import numpy as np
import random as rd

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from operation import *
from data_loader import *

device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=128, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=50, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=100, type=int, help='number of epochs')
args = parser.parse_args()

args = parser.parse_args()

class Cell(nn.Module):
    def __init__(self, out_dim=10, num_node=4, reduction=0, channel=3):
        super(Cell, self).__init__()
        stride = 2 if reduction==1  else 1
        
        self.op_0_1 = nn.DataParallel(OPS['sep_conv_3x3'](channel, stride, affine=False))     
        self.op_0_2 = nn.DataParallel(OPS['sep_conv_7x7'](channel, stride, affine=False))
        self.op_0_3 = nn.DataParallel(OPS['sep_conv_3x3'](channel, stride, affine=False))
        self.op_1_2 = nn.DataParallel(OPS['sep_conv_3x3'](channel, stride, affine=False))
        self.op_1_3 = nn.DataParallel(OPS['max_pool_3x3'](channel, stride, affine=False))
        self.op_2_3 = nn.DataParallel(OPS['sep_conv_3x3'](channel, stride, affine=False))
        
        # self.classifier = nn.Linear(3 * 3 * 32 * 32, out_dim)

    def forward(self, X):
        s01 = self.op_0_1(X)
        s02 = self.op_0_2(X)
        s03 = self.op_0_3(X)
        s12 = self.op_1_2(s01)
        s13 = self.op_1_3(s01)
        
        s2 = s02 + s12
        s23 = self.op_2_3(s2)

        # cat_out = torch.cat([s03, s13, s23], dim=1)
        # f_out = F.relu(torch.flatten(cat_out, 1))
        # out = self.classifier(f_out)
        out = s03 + s13 + s23
        return out
    
class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
class LinearNN(nn.Module):
    def __init__(self, c):
        super(LinearNN, self).__init__()
        self.classifier = nn.DataParallel(nn.Sequential(
                Flatten(),
                nn.Linear(3 * 32 * 32, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 128),
                nn.ReLU(True),
                nn.Linear(128, c)
                ))
        
    def forward(self,X):
        X = self.classifier(X)
        return X

class PoolingNN(nn.Module):
    def __init__(self):
        super(PoolingNN, self).__init__()
        self.pool = nn.DataParallel(nn.Sequential(
                nn.ReLU(inplace=False),
                nn.MaxPool2d(3, stride=1, padding=1)
                ))
        
    def forward(self,X):
        X = self.pool(X)
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
        torch.save(state, './checkpoint/cell'+str(index)+'.t'+str(i))
    best_acc = correct
    
    



class MyModule(nn.Module):
    def __init__(self, num_cell):
        super(MyModule, self).__init__()
        self.cells = nn.ModuleList([Cell(channel=3) for i in range(num_cell)])

    def forward(self, X):
        for i, l in enumerate(self.cells):
            X = self.cells[i](X)
        
        return X
    

# Main Function 
if __name__ == "__main__": 

    # task 0, 1, 22, 30
    base_task_list = np.array([ [1,3,8], [3,8,9], [2,6,7], [10] ], dtype=object)
    
    idx = 2
    
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
    torch.manual_seed(0)
    pool = PoolingNN().cuda()
    
    torch.manual_seed(0)
    linear = LinearNN(4).cuda()
    
    torch.manual_seed(0)
    net = MyModule(num_cell=50).cuda()
    
    model = torch.nn.Sequential(net, pool, linear)
    print(model)
    
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    
    # train and evaluate on the indicator task
    fit(model, trainloader)
    evaluate(model, testloader, test_label, True, idx, 0)
    
    
    
    
    
    