# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:30:50 2020

@author: catpl
"""

'''
This code is used to train the default network for the baseline/incoming tasks.
The baseline tasks:
    0/ MNIST - indicator digit 0
    1/ MNIST - indicator digit 1
    2/ MNIST - indicator digit 2
    3/ MNIST - indicator digit 3
    4/ MNIST - indicator digit 4
    5/ fMNIST - indicator object 0
    6/ fMNIST - indicator object 1
    7/ fMNIST - indicator object 2
    8/ fMNIST - indicator object 3
    9/ fMNIST - indicator object 4

Target task:
    0/ Quick, Draw! - indicator 
'''

import os
import argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from data_loader import *

class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=1, type=int, help='number of epochs')
args = parser.parse_args()

# Dataset
if not os.path.exists('./feature_data'):
    os.mkdir('./feature_data')


# Convolutional Neural Network Architecture for binary/multi-classification
class NN(nn.Module):
    def __init__(self, c):
        super(NN, self).__init__()
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
        torch.save(state, './checkpoint/task'+str(index)+'.t1')
    best_acc = correct


# Exstract features
def extract_feature(model, args, dataloader):
    print('==> start extracting features:')    
    with torch.no_grad():
        # Extract Feature
        model.eval()
        features = []
        
        for index, (inputs,_) in enumerate(dataloader):
            inputs = inputs.to(device)
            feature = model.encoder(inputs).cpu().detach().numpy()
            features.append(feature)
            
        features = np.concatenate(features,axis=0)
        print("Size of extracted features:")
        print(features.shape)
 
    return features

# -------------
# main code
# -------------
# task id
taskID = 101

# dataset = 'MNIST'
# dataset = 'fMNIST'
dataset = 'quickdraw'

# quickdraw! class object
class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']

indicator_idx = 6
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

# initialize CNN
torch.manual_seed(0)
cnn = NN(out_dim).cuda()
print(cnn)

# train and evaluate on the indicator task
fit(cnn, trainloader)
evaluate(cnn, testloader, test_label, True, taskID)

# # extract last-layer features
# train_feature = extract_feature(cnn, args, trainloader)
# test_feature = extract_feature(cnn, args, testloader)

# #save features
# print("Saving extracted features...")
# np.save('feature_data/task'+str(taskID)+'_train_feature_ind.npy', train_feature)
# np.save('feature_data/task'+str(taskID)+'_test_feature_ind.npy', test_feature)
# # save labels
# print("Saving train and test labels")
# np.save('feature_data/task'+str(taskID)+'_train_label_ind.npy', train_label)
# np.save('feature_data/task'+str(taskID)+'_test_label_ind.npy', test_label)

#load features
#feature = np.load('feature_data/feature_ind'+str(taskID)+'.npy')
