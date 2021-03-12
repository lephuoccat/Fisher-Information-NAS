# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:51:31 2020

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

from utils.masked_layer import MaskedLinear
from utils.pruning import weight_prune, prune_rate, finetune
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
parser.add_argument('--batch-size-train', default=200, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()

# Fine-tuning Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128, 
    'test_batch_size': 100,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'epoch':10
}


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


def load_feature(source, target, dataset, indicator_idx):
    # load dataset
    total_class = 10        # consider all 10 classes of data in MNIST/FashionMNIST/quickdraw
    out_dim = 2             # output dimension of CNN
    # quickdraw! class object
    class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']

    # load the train and test data for indicator task
    trainloader, testloader, train_label, test_label = indicator_dataset(dataset, indicator_idx, total_class, class_object, args)
    
    # Load CNN network source
    net_source = CNN(out_dim)
    net_source = net_source.to(device)
    checkpoint = torch.load('./checkpoint/task'+str(source)+'.t1')
    net_source.load_state_dict(checkpoint['net'])
    
    train_feature = extract_feature(net_source, args, trainloader)
    test_feature = extract_feature(net_source, args, testloader)


    # Load CNN network target
    net_target = CNN(out_dim)
    net_target = net_target.to(device)
    checkpoint = torch.load('./checkpoint/task'+str(target)+'.t1')
    net_target.load_state_dict(checkpoint['net'])
    
    train_target = extract_feature(net_target, args, trainloader)
    test_target = extract_feature(net_target, args, testloader)

    return train_feature, test_feature, train_target, test_target, train_label, test_label

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


transform = transforms.ToTensor()
class MNIST_feature_Dataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(1024, 1)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


# Transform neural network with mask
class NN(nn.Module):  
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = MaskedLinear(1024, 1024*2)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(1024*2, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(512, 1024)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
        


# Binary classification network
class binary_NN(nn.Module):
    def __init__(self):
        super(binary_NN, self).__init__()
        self.classifier = nn.Linear(1024,2)
        
    def forward(self,x):
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

# Train the CNN with MSE loss
def MSE_fit(model, train_loader, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    error = nn.MSELoss()
    EPOCHS = args.num_epoch
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total loss
            total_loss += loss.item()
        # print("epoch : {}/{}, MSE loss = {:.6f}".format(epoch + 1, EPOCHS, total_loss/len(label)))


# Test the CNN with MSE loss
def MSE_evaluate(model, test_loader, label):
    error = nn.MSELoss()
    total_loss = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = test_imgs.to(device)
        test_labels = test_labels.to(device)
        
        output = model(test_imgs)
        loss = error(output, test_labels)
        total_loss += loss.item()
    print("Test MSE loss = {:.6f}".format(total_loss/len(label)))
    
    
# Train CNN with cross-entropy loss
def CE_fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    error = nn.CrossEntropyLoss()
    EPOCHS = 50
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
            # if batch_idx % 50 == 0:
            #     print('Epoch : {} ({:.0f}%)\t Accuracy:{:.3f}%'.format(
            #         epoch, 100.*batch_idx / len(train_loader), float(correct*100) / float(args.batch_size_train*(batch_idx+1))))


# Test the CNN with cross-entropy loss
def CE_evaluate(model, test_loader, label):
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = test_imgs.to(device)
        test_labels = test_labels.to(device)
        
        output = model(test_imgs)
        predicted = torch.max(output.data,1)[1]
        correct += (predicted == test_labels).sum()
    print(correct)
    print("Test accuracy:{:.3f}% \n".format( float(correct * 100) / len(label)))
    
    
# Count non-zero weight
def countNonZeroWeights(model):
    nonzeros = 0
    allpara = 0
    for param in model.parameters():
        if param is not None:
            allpara += param.numel()
            nonzeros += param.nonzero().size(0)
    return allpara, nonzeros


# Exstract features
def extract_feature_last(model, args, dataloader):
    print('==> start extracting features:')    
    with torch.no_grad():
        # Extract Feature
        model.eval()
        features = []
        
        for index, (inputs,_) in enumerate(dataloader):
            inputs = inputs.to(device)
            feature = model(inputs).cpu().detach().numpy()
            features.append(feature)
            
        features = np.concatenate(features,axis=0)
        print("Size of extracted features:")
        print(features.shape)
 
    return features


# -------------
# main code
# -------------
# dataset = 'MNIST'
# dataset = 'fMNIST'
dataset = 'quickdraw'

# source and target task ID
source = 9
target = 101
indicator = 6
# load features and truth label
train_feature, test_feature, train_target, test_target, train_label, test_label = load_feature(source, target, dataset, indicator)


# create the dataset for transform network
trainset = MNIST_feature_Dataset(train_feature, train_target, transform)
testset = MNIST_feature_Dataset(test_feature, test_target, transform)
# create loader
trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)

# transform CNN
torch.manual_seed(0)
transform_cnn = NN().cuda()
print(transform_cnn)

# train and evaluate
MSE_fit(transform_cnn, trainloader, train_target)
MSE_evaluate(transform_cnn, testloader, test_target)

[all_param, non_zero] = countNonZeroWeights(transform_cnn)
print("All parameters from transform network: {}".format(all_param))
print("Non-zero parameters from transform network: {}".format(non_zero))







# Analyze the trained transform network
# create loader
trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=False)
testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
# extract features
output_train_feature = extract_feature_last(transform_cnn, args, trainloader)
output_test_feature = extract_feature_last(transform_cnn, args, testloader)

# create the dataset for classification network
traindata = MNIST_feature_Dataset(output_train_feature, train_label, transform)
testdata = MNIST_feature_Dataset(output_test_feature, test_label, transform)
# create loader
traindataloader = DataLoader(traindata, batch_size=args.batch_size_train, shuffle=True)
testdataloader = DataLoader(testdata, batch_size=args.batch_size_test, shuffle=False)

# binary classification network
torch.manual_seed(0)
binary_cnn = binary_NN().cuda()
print(binary_cnn)

# train and evaluate
CE_fit(binary_cnn, traindataloader)
CE_evaluate(binary_cnn, testdataloader, test_label)





'''
# prune the weights
masks = weight_prune(transform_cnn, 5)
transform_cnn.set_masks(masks)
print("Begin pruning...")

# Retraining
finetune(transform_cnn, param, trainloader)
prune_rate(transform_cnn)

# Test
# Analyze the trained transform network
trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=False)
testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
# extract features
output_train_feature = extract_feature_last(transform_cnn, args, trainloader)
output_test_feature = extract_feature_last(transform_cnn, args, testloader)

# create the dataset for classification network
traindata = MNIST_feature_Dataset(output_train_feature, train_label, transform)
testdata = MNIST_feature_Dataset(output_test_feature, test_label, transform)
# create loader
traindataloader = DataLoader(traindata, batch_size=args.batch_size_train, shuffle=True)
testdataloader = DataLoader(testdata, batch_size=args.batch_size_test, shuffle=False)

# binary classification network
torch.manual_seed(0)
binary_cnn = binary_NN().cuda()
print(binary_cnn)

# train and evaluate
CE_fit(binary_cnn, traindataloader)
CE_evaluate(binary_cnn, testdataloader, test_label)

'''