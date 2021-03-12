# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:36:38 2021

@author: catpl
"""
import os
import argparse

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import torchvision.models as models
from models import *
from torch.utils.data import Dataset, DataLoader
from skimage import io
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations 
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
# transform = transforms.Compose([transforms.Resize((256, 256)), 
#                     transforms.ToTensor()])

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
        # target_image = target_image.astype(np.float32)
        # target_image = np.tile(target_image[:, :, None], [1, 1, 3])
        
        target_image = Image.fromarray(target_image.astype('uint8'), 'RGB')
        
        
        sample = {'source': source_image, 'target': target_image}
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        sample = {'source': source_image, 'target': target_image}
        
        return sample


def show_landmarks(source, target):
    print(source)
    print(target)
    plt.figure(0)
    plt.imshow(torchvision.utils.make_grid(target, nrow=5).permute(1, 2, 0))
    plt.figure(1)
    plt.imshow(torchvision.utils.make_grid(source, nrow=5).permute(1, 2, 0))

# for i in range(len(my_dataset)):
#     sample = my_dataset[i]
#     print(i, sample['source'].shape, sample['target'].shape)
#     show_landmarks(sample['source'], sample['target'])
#     if i == 0:
#         break


class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
    
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
        # self.encoder = nn.DataParallel(self.encoder)
                
        self.decoder = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(True),
                nn.Linear(512, 3 * 128 * 128),
                nn.ReLU(True))
        # self.decoder = nn.DataParallel(self.decoder)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)
        X = X.view(X.size(0), 3, 128, 128)
        return X


# Train the CNN
def fit(model, train_loader, save_flag, i):
    optimizer = torch.optim.Adam(model.parameters())
    # error = nn.CrossEntropyLoss()
    error = nn.MSELoss()
    EPOCHS = args.num_epoch
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            inputs = resize2d(data['source'],  (128,128))
            inputs = inputs.to(device)
            targets = resize2d(data['target'],  (128,128))
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total loss
            total_loss += loss.item()
            # print("batch : {}, loss = {:.6f}".format(batch_idx, total_loss))
            
        total_loss = total_loss / len(train_loader)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, total_loss))
        
    if (save_flag == True):
        # Save the pretrained network
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': total_loss,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/taskonomy_principal_curvature.t'+str(i))

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

# Main Function 
if __name__ == "__main__": 
    for i in range(5):
        my_dataset = CustomDataSet('./data/taskonomy/rgb/', './data/taskonomy/principal_curvature/', transform=transform)
        trainloader = DataLoader(my_dataset , batch_size=args.batch_size_train, shuffle=True)

        print('Length of dataset: ')
        print(len(my_dataset))
        
        # for i in range(len(my_dataset)):
        #     sample = my_dataset[i]
        #     sample['source']= resize2d(sample['source'],  (128,128))
        #     sample['target']= resize2d(sample['target'],  (128,128))
        #     print(i, sample['source'].shape, sample['target'].shape)
        #     show_landmarks(sample['source'], sample['target'])
        #     if i == 0:
        #         break

        # initialize CNN
        torch.manual_seed(i)
        model = NN().cuda()
        print(model)
            
        # train and evaluate on the indicator task
        fit(model, trainloader, True, i)
        
        
        
