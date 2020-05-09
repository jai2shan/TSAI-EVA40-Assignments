# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:31:19 2020

@author: jayasans4085
"""
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class DataTransformation:
    torch.manual_seed(1)

    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)
        if self.cuda:
            torch.cuda.manual_seed(1)

    def TrainPrep(self):
        # Train Phase transformations
        train_transforms = transforms.Compose([
                                              #  transforms.Resize((28, 28)),
                                              #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                               transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                               # Note the difference between (0.1307) and (0.1307,)
                                               ])
    
        train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
        # train dataloader
        train_loader = torch.utils.data.DataLoader(train, **self.dataloader_args)
        return train_loader
    
    def TestPrep(self):
        # Test Phase transformations
        test_transforms = transforms.Compose([
                                              #  transforms.Resize((28, 28)),
                                              #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                               ])
        test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
        # test dataloader
        test_loader = torch.utils.data.DataLoader(test, **self.dataloader_args)
        return test_loader