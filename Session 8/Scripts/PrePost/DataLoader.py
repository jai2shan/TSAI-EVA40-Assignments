# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:45:51 2020

@author: jayasans4085
"""
import torch
from torchvision import datasets
import torchvision

import matplotlib.pyplot as plt
import numpy as np

def cifar_data_loader(transform_params,BatchSize):
    trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_params)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize,
                                              shuffle=True, num_workers=4)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_params)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize,
                                             shuffle=False, num_workers=4)
    return trainloader,testloader

def View_images(trainloader,classes):
    # functions to show an image
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images[:4]))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
