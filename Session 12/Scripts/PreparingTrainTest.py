# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:48:51 2020

@author: ***
"""

import os
os.chdir(r'C:\Users\jayasans4085\OneDrive - ***\Desktop\TSAI Sessions\tiny-imagenet-200\tiny-imagenet-200\train')
basepath = r'C:\Users\jayasans4085\OneDrive - ***\Desktop\TSAI Sessions\tiny-imagenet-200 - Copy\tiny-imagenet-200\train'
import random
import shutil
#train=data.sample(frac=0.8,random_state=200)
j = 0
train = r'C:\Users\jayasans4085\OneDrive - ***\Desktop\TSAI Sessions\tiny-imagenet-200 - Copy\tiny-imagenet-200\AssignmentData\train'
test = r'C:\Users\jayasans4085\OneDrive - ***\Desktop\TSAI Sessions\tiny-imagenet-200 - Copy\tiny-imagenet-200\AssignmentData\test'

classes = dict()
j = 0

os.chdir(basepath)
for filename in os.listdir():
    classes[j] = filename
    
    os.mkdir(train+'\\'+str(j))
    os.mkdir(test+'\\'+str(j))
    os.chdir(basepath+'\\'+filename+'\\'+'images')
    testfiles = random.sample(os.listdir(), 150)

    for i in os.listdir():
        if i in testfiles:
            shutil.move(i,test+'\\'+str(j)+'\\'+i)
        else:
            shutil.move(i,train+'\\'+str(j)+'\\'+i)
    print(j)
    j = j+1
    os.chdir(basepath)

import pandas as pd
classes = pd.DataFrame(classes.items(),columns = ['Folder Names','Class Names'])
classes.to_csv('Classes.csv')
