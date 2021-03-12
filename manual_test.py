# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:22:52 2020

@author: catpl
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from data_loader import feature_Dataset
from operation import *

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=50, type=int, help='number of epochs')
args = parser.parse_args()


model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model_resnet34 = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
model_densenet161 = torch.hub.load('pytorch/vision', 'densenet161', pretrained=True)

for name, param in model_resnet18.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
        
for name, param in model_resnet34.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

for name, param in model_densenet161.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
        
for x in model_densenet161.modules():
    if isinstance(x, nn.AvgPool2d):
        x.ceil_mode = True
        
num_classes = 2

# model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,512),
#                                   nn.ReLU(),
#                                   nn.Dropout(),
#                                   nn.Linear(512, num_classes))

# model_resnet34.fc = nn.Sequential(nn.Linear(model_resnet34.fc.in_features,512),
#                                   nn.ReLU(),
#                                   nn.Dropout(),
#                                   nn.Linear(512, num_classes))

model_densenet161.classifier = nn.Sequential(nn.Linear(model_densenet161.classifier.in_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3,-1,-1))
    ])

transforms = transforms.ToTensor()

def indicator_dataset(dataset, num, limit, class_object, args):
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
    trainset = feature_Dataset(train_data, train_label.astype(int), transforms)
    trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
    
    testset = feature_Dataset(test_data, test_label.astype(int), transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.encoder = nn.Sequential(
#                 nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(True),
#                 nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(True),
#                 nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),
#                 nn.MaxPool2d(2, stride=2, padding=1),
#                 nn.ReLU(True),
#                 Flatten(),
#                 nn.Linear(27 * 27 * 5, 700),
#                 nn.ReLU(True),
#                 nn.Linear(700, 500),
#                 nn.ReLU(True),
#                 nn.Linear(500, 400),
#                 nn.ReLU(True),
#                 nn.Linear(400, 345),
#                 nn.ReLU(True))
#         # c is the number of class label
#         self.classifier = nn.Linear(345,2)
        
#     def forward(self,X):
#         X = self.encoder(X)
#         X = self.classifier(X)
#         return X


class Cell(nn.Module):
    def __init__(self, dim, num_node=3, reduction=0, channel=1):
        super(Cell, self).__init__()
        stride = 2 if reduction==1  else 1
        
        self.op_0_1 = OPS['sep_conv_3x3'](channel, stride, affine=False)
        self.op_0_2 = OPS['sep_conv_3x3'](channel, stride, affine=False)
        self.op_1_2 = OPS['dil_conv_5x5'](channel, stride, affine=False)
              
        self.classifier = nn.Linear(2 * 1 * 28 * 28, dim)

    def forward(self, X):
        s01 = self.op_0_1(X)
        s02 = self.op_0_2(X)
        s12 = self.op_1_2(s01)

        # out = s02 + s12
        cat_out = torch.cat([s02, s12], dim=1)
        f_out = F.relu(torch.flatten(cat_out, 1))
        out = self.classifier(f_out)
        return out





def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):
    print("Training...")
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.long().to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
                        
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))


def test_model(model, test_data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))
    
    
batch_size=32
img_dimensions = 224


# Load data
dataset = 'quickdraw'
class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']
indicator_idx = 6
total_class = 10  

train_data_loader, test_data_loader, train_label, test_label = indicator_dataset(dataset, indicator_idx, total_class, class_object, args)




if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

# ResNet18
# model_resnet18.to(device)
# optimizer = optim.Adam(model_resnet18.parameters(), lr=0.001)
# train(model_resnet18, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, train_data_loader, epochs=50, device=device)
# test_model(model_resnet18, test_data_loader)

# ResNet34
# model_resnet34.to(device)
# optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)
# train(model_resnet34, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, train_data_loader, epochs=50, device=device)
# test_model(model_resnet34, test_data_loader)

# DenseNet161
# model_densenet161.to(device)
# optimizer = optim.Adam(model_densenet161.parameters(), lr=0.001)
# train(model_densenet161, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, train_data_loader, epochs=20, device=device)
# test_model(model_densenet161, test_data_loader)

# CNN from paper
# CNN = NN().to(device)
# optimizer = optim.Adam(CNN.parameters(), lr=0.001)
# train(CNN, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, train_data_loader, epochs=50, device=device)
# test_model(CNN, test_data_loader)

# CNN from OPS
torch.manual_seed(0)
CNN = Cell(2).to(device)
optimizer = optim.Adam(CNN.parameters(), lr=0.001)
train(CNN, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, train_data_loader, epochs=1, device=device)
test_model(CNN, test_data_loader)






def countNonZeroWeights(model):
    nonzeros = 0
    allpara = 0
    for param in model.parameters():
        if param is not None:
            allpara += param.numel()
            nonzeros += param.nonzero().size(0)
    return allpara, nonzeros


[all_param, non_zero] = countNonZeroWeights(CNN)
print("All parameters from transform network: {}".format(all_param))
print("Non-zero parameters from transform network: {}".format(non_zero))