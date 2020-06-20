# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:04:42 2020

@author: jayasans4085
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,num_splits_=1):
        super(Net, self).__init__()
        dropout_value = 0.1
        # Prep Layer
        self.prep = nn.Sequential(
                                    ## Convolution 1
                                    nn.Conv2d(in_channels=3, 
                                              out_channels=64, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.ReLU(),
                                    GhostBatchNorm(64)
                                    )
        # Layer 1
        self.l1x = nn.Sequential(
                                    nn.Conv2d(in_channels=64, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.MaxPool2d(2, 2),
                                    GhostBatchNorm(128),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_value)
                                )
        self.l1r1 = nn.Sequential(
                                    nn.Conv2d(in_channels=128, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    GhostBatchNorm(128),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=128, 
                                              out_channels=128, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    GhostBatchNorm(128),
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
                                GhostBatchNorm(256),
                                nn.ReLU(),
                                nn.Dropout(dropout_value)
                                )
        
        # Layer 3
        self.l3x = nn.Sequential(
                                    nn.Conv2d(in_channels=256, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    nn.MaxPool2d(2, 2),
                                    GhostBatchNorm(512),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_value)  
                                )
        self.l3r2 = nn.Sequential(
                                    nn.Conv2d(in_channels=512, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    GhostBatchNorm(512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, 
                                              out_channels=512, 
                                              kernel_size=(3, 3), 
                                              padding=1, 
                                              bias=False),
                                    GhostBatchNorm(512),
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
        x = self.l1x(x)
        x = x + self.l1r1(x)
        # Layer 2
        x = self.l2(x)
        # Layer 3
        x = self.l3x(x)
        x = x+self.l3r2(x)
        # Output
        x = self.output(x)

        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)
