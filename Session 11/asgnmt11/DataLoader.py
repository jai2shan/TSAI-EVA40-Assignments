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

class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, prob, n_holes, length):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        # if np.random.random()>self.prob:
        #   return img
        # else:

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

             mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

          return img

def cifar_data_loader(BatchSize):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform_params = dict()
    transform_params['train'] = transforms.Compose([
                                       transforms.RandomRotation(10),
                                       transforms.RandomHorizontalFlip(),  
                                       transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)), 
                                       transforms.RandomCrop(size = 32,padding = 4),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16,prob = 0.5),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    transform_params['test'] = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_params['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize,
                                              shuffle=True, num_workers=4)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_params['test'])
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
