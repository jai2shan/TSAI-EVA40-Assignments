# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:41:05 2020

@author: jayasans4085
"""

import torch
import torchvision
import os
from torchvision import transforms
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Scripts')

from DataLoader import *
from torchvision import transforms

import torch
from torchsummary import summary
from resnet import ResNet18

from LR_Finder import *
from Training_Testing import TrainTest
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import pandas as pd

train_path = r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Data\train'
test_path = r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Data\test'
train_loader,test_loader = data_loader_(train_path,test_path,128)

#%%
import torch
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Scripts')
from resnet import ResNet18
net = ResNet18(num_classes = 200).to(device)
summary(net, input_size=(3,32,32))

#%%
tt = TrainTest()
test_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

lr_finder = LRFinder(net, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=500,step_mode="exp")
lr_finder.plot()

lr_ = pd.DataFrame(lr_finder.history)
lr_max = lr_.loc[lr_['loss']==lr_['loss'].min(),'lr'].values[0]
lr_max
#%%
net = ResNet18(num_classes = 200)).to(device)
tt = TrainTest()
test_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_max, momentum=0.9,weight_decay = 0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

for epoch in range(50):
    print("EPOCH:", epoch)
    tt.train_(net, device, trainloader, optimizer, criterion, epoch,L1 = False)
    acc = tt.test_(net, device, testloader)  
    scheduler.step(tt.test_losses[epoch])
       
print('Finished Training')