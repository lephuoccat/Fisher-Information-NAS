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
parser.add_argument('--batch-size-train', default=128, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=100, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=100, type=int, help='number of epochs')
args = parser.parse_args()  

# Convolutional Neural Network Architecture for binary/multi-classification
class BinaryNN(nn.Module):
    def __init__(self, c):
        super(BinaryNN, self).__init__()
        self.classifier = nn.Linear(10,c)
        
    def forward(self,X):
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
        torch.save(state, './checkpoint/cifar100_task'+str(index)+'.t'+str(i))
    best_acc = correct


# Main Function 
if __name__ == "__main__": 
    for i in range(10):
        
        # task 1: (binary) 18: [8, 58, 90, 13, 48] 19: [81, 69, 41, 89, 85]
        # task 2: (binary) 5: [40, 39, 22, 87, 86] 6: [20, 25, 94, 84, 5]
        # task 3: (multi)
        # task 4: (all)
        base_task_list = np.array([ [8, 58, 90, 13, 48, 81, 69, 41, 89, 85], 
                                   [40, 39, 22, 87, 86, 20, 25, 94, 84, 5], 
                                   [8, 58, 90, 13, 48, 81, 69, 41, 89, 85], 
                                   [8, 58, 90, 13, 48, 81, 69, 41, 89, 85, 40, 39, 22, 87, 86, 20, 25, 94, 84, 5] ], dtype=object)

        # load CIFAR10 indicator dataset
        for idx in range(len(base_task_list)):
            print(base_task_list[idx])

            if idx == 2:
                print('Loading multi-class dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset('CIFAR100', base_task_list[idx], args)
                c = 11
            elif idx == 3:
                print('Loading 10-class dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset('CIFAR100', base_task_list[idx], args)
                c = 21
            else:
                print('Loading binary dataset...')
                trainloader, testloader, train_label, test_label = CIFAR_indicator_dataset('CIFAR100', base_task_list[idx], args)
                c = 2
            
            
            # initialize CNN
            torch.manual_seed(i)
            net = VGG('VGG16').cuda()
            # net = ResNet18().cuda()
            # net = DenseNet121().cuda()
            
            torch.manual_seed(i)
            binary = BinaryNN(c).cuda()
            
            model = torch.nn.Sequential(net, binary)
            print(model)
            
            
            
            # train and evaluate on the indicator task
            fit(model, trainloader)
            evaluate(model, testloader, test_label, True, idx, i)
        
        
        
        
        
        
        
        
'''
0: [72, 4, 95, 30, 55]
1: [73, 32, 67, 91, 1]
2: [92, 70, 82, 54, 62]
3: [16, 61, 9, 10, 28]
4: [51, 0, 53, 57, 83]
5: [40, 39, 22, 87, 86]
6: [20, 25, 94, 84, 5]
7: [14, 24, 6, 7, 18]
8: [43, 97, 42, 3, 88]
9: [37, 17, 76, 12, 68]
10: [49, 33, 71, 23, 60]
11: [15, 21, 19, 31, 38]
12: [75, 63, 66, 64, 34]
13: [77, 26, 45, 99, 79]
14: [11, 2, 35, 46, 98]
15: [29, 93, 27, 78, 44]
16: [65, 50, 74, 36, 80]
17: [56, 52, 47, 59, 96]
18: [8, 58, 90, 13, 48]
19: [81, 69, 41, 89, 85]

Sub classes for Super class  0: seal,beaver,whale,dolphin,otter,
Sub classes for Super class  1: shark,flatfish,ray,trout,aquarium_fish,
Sub classes for Super class  2: tulip,rose,sunflower,orchid,poppy,
Sub classes for Super class  3: can,plate,bottle,bowl,cup,
Sub classes for Super class  4: mushroom,apple,orange,pear,sweet_pepper,
Sub classes for Super class  5: lamp,keyboard,clock,television,telephone,
Sub classes for Super class  6: chair,couch,wardrobe,table,bed,
Sub classes for Super class  7: butterfly,cockroach,bee,beetle,caterpillar,
Sub classes for Super class  8: lion,wolf,leopard,bear,tiger,
Sub classes for Super class  9: house,castle,skyscraper,bridge,road,
Sub classes for Super class  10: mountain,forest,sea,cloud,plain,
Sub classes for Super class  11: camel,chimpanzee,cattle,elephant,kangaroo,
Sub classes for Super class  12: skunk,porcupine,raccoon,possum,fox,
Sub classes for Super class  13: snail,crab,lobster,worm,spider,
Sub classes for Super class  14: boy,baby,girl,man,woman,
Sub classes for Super class  15: dinosaur,turtle,crocodile,snake,lizard,
Sub classes for Super class  16: rabbit,mouse,shrew,hamster,squirrel,
Sub classes for Super class  17: palm_tree,oak_tree,maple_tree,pine_tree,willow_tree,
Sub classes for Super class  18: bicycle,pickup_truck,train,bus,motorcycle,
Sub classes for Super class  19: streetcar,rocket,lawn_mower,tractor,tank,
'''
        
        
        
        
        
        
        
        
        
        
        