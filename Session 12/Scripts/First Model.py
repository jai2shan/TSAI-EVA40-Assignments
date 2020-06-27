# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:56:10 2020

@author: jayasans4085
"""

import torch
import torchvision
import os
from torchvision import transforms
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Documents\Neural Networks')

def load_dataset(data_path):
    
    data_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor()  
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=True
    )
    return data_loader

data_path = r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Data\train'
train_loader = load_dataset(data_path)

data_path = r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI-EVA40-Assignments\Session 12\Data\test'
test_loader = load_dataset(data_path)

#%%
import torch
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

#os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Documents\Neural Networks\Codes\Models')
#from resnet import ResNet18
net = ResNet18(num_classes = 200).to(device)
summary(net, input_size=(3,32,32))


#%%
#from Training_Testing import TrainTest
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

tt = TrainTest()
test_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay = 0.00005)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(30):
    print("EPOCH:", epoch)
    tt.train_(net, device, train_loader, optimizer, criterion, epoch,L1 = False)
    scheduler.step()
    acc = tt.test_(net, device, test_loader)  
    
   
print('Finished Training')