# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:43:01 2021

@author: catpl
"""
import os
import argparse

import numpy as np
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from skimage import io

from models import *
from data_loader import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='NAS Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=512, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=50, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()

transform = transforms.ToTensor()

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=False), size)).data

class CustomDataSet(Dataset):
    def __init__(self, source_dir, target_dir, transform):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.source_imgs = os.listdir(source_dir)
        self.target_imgs = os.listdir(target_dir)

    def __len__(self):
        return len(self.source_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        source_img_loc = os.path.join(self.source_dir, self.source_imgs[idx])
        source_image = io.imread(source_img_loc)
        
        target_img_loc = os.path.join(self.target_dir, self.target_imgs[idx])
        target_image = io.imread(target_img_loc)
        # preprocess target image to 3 channel
        target_image = target_image.astype(np.float32)
        target_image = np.tile(target_image[:, :, None], [1, 1, 3])
        
        target_image = Image.fromarray(target_image.astype('uint8'), 'RGB')
        
        
        sample = {'source': source_image, 'target': target_image}
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        sample = {'source': source_image, 'target': target_image}
        
        return sample











class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

# Convolutional Neural Network Architecture for binary/multi-classification
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.encoder = nn.Sequential(
                Flatten(),
                nn.Linear(3 * 128 * 128, 512),
                nn.ReLU(True),
                nn.Linear(512, 128),
                nn.ReLU(True))
                
                
        # c is the number of class label
        self.decoder = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(True),
                nn.Linear(512, 3 * 128 * 128),
                nn.ReLU(True))
        
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)
        X = X.view(X.size(0), 3, 128, 128)
        return X



def diag_fisher(model, data):
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)
    
    model.eval()
    error = nn.MSELoss()
    for batch_idx, data in enumerate(data):
        inputs = resize2d(data['source'],  (128,128))
        inputs = inputs.to(device)
        targets = resize2d(data['target'],  (128,128))
        targets = targets.to(device)
        
        model.zero_grad()
        output = model(inputs)

        loss = error(output, targets)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    
    return precision_matrices


# List of task
# 0: depth_euclidean
# 1: depth_zbuffer
# 2: edge_occlusion
# 3: edge_texture
# 4: keypoints2d
# 5: keypoints3d
# 6: normal
# 7: principal_curvature
# 8: reshading

# Main code
my_dataset = CustomDataSet('./data/taskonomy/rgb/', './data/taskonomy/depth_euclidean/', transform=transform)
trainloader = DataLoader(my_dataset , batch_size=args.batch_size_train, shuffle=False)

# file_idx = 0
for file_idx in range(5):
    print(file_idx)
    # --------
    # Source
    # --------
    net_source = NN().cuda()
    checkpoint = torch.load('./checkpoint/taskonomy_depth_zbuffer.t'+str(file_idx))
    net_source.load_state_dict(checkpoint['net'])
    
    # calculate fisher matrix
    fisher_matrix_source = diag_fisher(net_source, trainloader)
    
    total_source = 0
    for n, p in net_source.named_parameters():
        total_source += np.sum(fisher_matrix_source[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in net_source.named_parameters():
        fisher_matrix_source[n] = fisher_matrix_source[n]/total_source
    
    # --------
    # Target
    # --------
    net_target = NN().cuda()
    checkpoint = torch.load('./checkpoint/taskonomy_depth_euclidean.t'+str(file_idx))
    net_target.load_state_dict(checkpoint['net'])
    
    # calculate fisher matrix
    fisher_matrix_target = diag_fisher(net_target, trainloader)
    
    total_target = 0
    for n, p in net_target.named_parameters():
        total_target += np.sum(fisher_matrix_target[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in net_target.named_parameters():
        fisher_matrix_target[n] = fisher_matrix_target[n]/total_target
    
    fisher_matrix_target = fisher_matrix_source
    # Frechet distance
    distance = 0
    for n, p in net_source.named_parameters():
        distance += 0.5 * np.sum(((fisher_matrix_source[n]**0.5 - fisher_matrix_target[n]**0.5)**2).cpu().numpy())
    
    print(distance)

    
