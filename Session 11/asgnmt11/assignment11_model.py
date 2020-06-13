# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:04:42 2020

@author: jayasans4085
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Prep Layer
        self.prep = nn.Sequential(
                                    ## Convolution 1
                                    nn.Conv2d(in_channels=3, 
                                              out_channels=64, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64)
                                    )
        # Layer 1
        self.l1x = nn.Sequential(
                                    nn.Conv2d(in_channels=64, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.MaxPool2d(2, 2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()     
                                )
        self.l1r1 = nn.Sequential(
                                    nn.Conv2d(in_channels=128, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=128, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()            
                                )
        # Layer 2
        self.l2 = nn.Sequential(
                                nn.Conv2d(in_channels=128, 
                                              out_channels=256, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                nn.MaxPool2d(2, 2),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                )
        
        # Layer 3
        self.l3x = nn.Sequential(
                                    nn.Conv2d(in_channels=256, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.MaxPool2d(2, 2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU()     
                                )
        self.l3r2 = nn.Sequential(
                                    nn.Conv2d(in_channels=512, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU()            
                                )
        self.output = nn.Sequential(
                                    nn.MaxPool2d(4, 2),
                                    nn.Conv2d(in_channels=512, 
                                              out_channels=10, 
                                              kernel_size=(1, 1), 
                                              padding=0, 
                                              bias=False)
                                    )
        
        
    def forward(self, x):
        x = self.prep(x)
        # Layer 1
        x = self.l1x(x)+self.l1r1(self.l1x(x))
        # Layer 2
        x = self.l2(x)
        # Layer 3
        x = self.l3x(x)+self.l3r2(self.l3x(x))
        # Output
        x = self.output(x)

        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)