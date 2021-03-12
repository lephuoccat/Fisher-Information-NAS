# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:21:38 2020

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
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=50, type=int, help='number of epochs')
parser.add_argument('--num-epoch-child', default=3, type=int, help='number of epochs for child training')
parser.add_argument('--num-epoch-coef', default=10, type=int, help='number of epochs for coef training')
parser.add_argument('--num-child', default=3, type=int, help='number of child networks')

args = parser.parse_args()

# Dataset
if not os.path.exists('./feature_data'):
    os.mkdir('./feature_data')
    
def weight_reset(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        model.reset_parameters()

def weight_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight.data)
        
# 3 nodes cell
# class Cell(nn.Module):
#     def __init__(self, dim, num_node=3, reduction=0, channel=3):
#         super(Cell, self).__init__()
#         stride = 2 if reduction==1  else 1
        
#         self.op_0_1 = OPS[draw_operation(9)](channel, stride, affine=False)
#         self.op_0_2 = OPS[draw_operation(9)](channel, stride, affine=False)
#         self.op_1_2 = OPS[draw_operation(9)](channel, stride, affine=False)
        
        
#         self.classifier = nn.Linear(2 * 1 * 28 * 28, dim)

#     def forward(self, X):
#         s01 = self.op_0_1(X)
#         s02 = self.op_0_2(X)
#         s12 = self.op_1_2(s01)

#         # out = s02 + s12
#         cat_out = torch.cat([s02, s12], dim=1)
#         f_out = F.relu(torch.flatten(cat_out, 1))
#         out = self.classifier(f_out)
#         return out

# # 4 nodes cell
class Cell(nn.Module):
    def __init__(self, dim, num_node=4, reduction=0, channel=3):
        super(Cell, self).__init__()
        stride = 2 if reduction==1  else 1
        
        self.op_0_1 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))     
        self.op_0_2 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))
        self.op_0_3 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))
        self.op_1_2 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))
        self.op_1_3 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))
        self.op_2_3 = nn.DataParallel(OPS[draw_operation(9)](channel, stride, affine=False))
        
        self.classifier = nn.Linear(3 * 3 * 32 * 32, dim)

    def forward(self, X):
        s01 = self.op_0_1(X)
        s02 = self.op_0_2(X)
        s03 = self.op_0_3(X)
        s12 = self.op_1_2(s01)
        s13 = self.op_1_3(s01)
        
        s2 = s02 + s12
        s23 = self.op_2_3(s2)

        cat_out = torch.cat([s03, s13, s23], dim=1)
        f_out = F.relu(torch.flatten(cat_out, 1))
        out = self.classifier(f_out)
        return out

# Train the CNN
def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
            if batch_idx % 200 == 0:
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
        torch.save(state, './checkpoint/bestnet'+str(index)+'.t1')
    best_acc = float(correct * 100) / len(label) 



# Train the child models
def child_train(coef, model1, model2, model3, trainloader):
    optimizer1 = torch.optim.Adam(model1.parameters())
    optimizer2 = torch.optim.Adam(model2.parameters())
    optimizer3 = torch.optim.Adam(model3.parameters())
    error = nn.CrossEntropyLoss()

    model1.train()
    model2.train()
    model3.train()
    for epoch in range(args.num_epoch_child):
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            
            output1 = model1(inputs) 
            output2 = model2(inputs)
            output3 = model3(inputs)
            output = (torch.exp(coef[0,0]) * output1 + torch.exp(coef[0,1]) * output2 + torch.exp(coef[0,2]) * output3) / torch.sum(torch.exp(coef))
            loss = error(output, targets)
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            
            loss.backward()
            
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            
            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == targets).sum()
            #print(correct)
            # if batch_idx % 200 == 0:
            #     print('Epoch : {} ({:.0f}%)\t Accuracy:{:.3f}%'.format(
            #         epoch, 100.*batch_idx / len(trainloader), float(correct*100) / float(args.batch_size_train*(batch_idx+1))))


# Train the coefficients
def coef_train(coef, model1, model2, model3, testloader):
    optimizer = torch.optim.Adam([coef])
    error = nn.CrossEntropyLoss()
    for epoch in range(args.num_epoch_coef):
        # correct = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            
            optimizer.zero_grad()
            output1 = model1(inputs) 
            output2 = model2(inputs)
            output3 = model3(inputs)
            output = (torch.exp(coef[0,0]) * output1 + torch.exp(coef[0,1]) * output2 + torch.exp(coef[0,2]) * output3)/torch.sum(torch.exp(coef))
    
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # if batch_idx % 50 == 0:
            #     print('Epoch : {} ({:.0f}%)\t coef:[{:.3f}, {:.3f}, {:.3f}]'.format(
            #         epoch, 100.*batch_idx / len(testloader), coef[0,0], coef[0,1], coef[0,2]))


# Generate and train child network
def FUSE(child1, dim, train_loader, test_loader, test_label):
    # reset best_model weights
    torch.manual_seed(0)
    child1.apply(weight_init)
    child1.to(device)
    # print("Child network 1:")
    # print(child1)
    # print("\n")
    
    # generate child 2 and 3
    torch.manual_seed(0)
    child2 = Cell(dim)
    child2.to(device)
    # print("Child network 2:")
    # print(child2)
    # print("\n")
    
    torch.manual_seed(0)
    child3 = Cell(dim)
    child3.to(device)
    # print("Child network 3:")
    # print(child3)
    # print("\n")
    child_network = [child1, child2, child3]
    
    # coefficient for child networks
    # coef = Variable(torch.ones(1, args.num_child), requires_grad=True)
    coef = torch.ones(1, args.num_child)
    coef = coef.to(device).requires_grad_(True)
    
    # jointly train child networks and coef
    for i in range(10):
        print("Training Stage: {}".format(i))
        print("Training child networks...")
        child_train(coef, child1, child2, child3, train_loader)
        print("Training coefficients...")
        coef_train(coef, child1, child2, child3, test_loader)
        print(coef)
        
    # Choose the best of child network
    values, indices = torch.max(coef, 1)
    current_best = child_network[indices]
    
    return current_best, indices



# ------------
# main file
# ------------
def main():
    torch.autograd.set_detect_anomaly(False)
    # task id
    taskID = 50
    
    dataset = 'CIFAR10'
    
    base_task_list = np.array([ [1,3,8], [3,8,9], [2,6,7], [10] ])
    target = 2
    out_dim = 4

    # load the train and test data for indicator task
    trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)

    # Train and evaluate the seed network
    torch.manual_seed(0)
    best_model = Cell(out_dim)
    best_model.to(device)
    print(best_model)
    # fit(best_model, trainloader)
    # evaluate(best_model, testloader, test_label, False, taskID)
    # print("The current best accuracy:{:.3f} \n".format( float(best_acc * 100) / len(test_label)))
    
    # Generate new child network
    for iter in range(100):
        print("NAS Stage: {}".format(iter))
        
        # Generate and choose the best child
        best_model, idx = FUSE(best_model, out_dim, trainloader, testloader, test_label)
        
        # if (idx != 0):
            # Train and evaluate the best child
            # fit(best_model, trainloader)
            # evaluate(best_model, testloader, test_label, False, taskID)
            
        # print("Best index: {}".format(idx))
        # print("The current best network:")
        # print(best_model)
        # print("\n")
        # print("------------------------------------------------------------")
        # print("\n")
        # print("The current best accuracy: {:.3f} \n".format(best_acc))
    
    
    
    # reset best_model
    torch.manual_seed(0)
    best_model.apply(weight_init)

    # train and evaluate
    fit(best_model, trainloader)
    evaluate(best_model, testloader, test_label, False, taskID)
    
    print("The best network:")
    print(best_model)
    print("\n")
    print("The current best accuracy: {:.3f} \n".format(best_acc))
    

if __name__ == "__main__":
    main()




