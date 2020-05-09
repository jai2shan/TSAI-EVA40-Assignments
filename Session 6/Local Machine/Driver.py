
from __future__ import print_function
import torch
import torch.optim as optim


import os
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI\Session 6\Modules')
from Model import Net
from DataTransformation import DataTransformation
from TrainTest import TrainTest

#%%
cuda = torch.cuda.is_available()
print("CUDA Available?", torch.cuda.is_available())

dt = DataTransformation()
# train dataloader
train_loader = dt.TrainPrep()
# test dataloader
test_loader = dt.TrainPrep()

#%%
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
#%%
from torch.optim.lr_scheduler import StepLR

models_ = {
            1:{'L1' : True,'L2' : True},
            2:{'L1' : True,'L2' : False},
            3:{'L1' : False,'L2' : True},
            4:{'L1' : False,'L2': False}
            }

for i in [1,2,3,4]:
    model =  Net().to(device)
    def Optimizer_(L2 = models_[i]['L2']):
        if (L2 == False):
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9,weight_decay = 0.00005)
        return optimizer
    optimizer = Optimizer_(L2 = True)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    tt = TrainTest()
    EPOCHS = 2
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        tt.train_(model, device, train_loader, optimizer, epoch,L1 = models_[i]['L1'])
        scheduler.step()
        tt.test_(model, device, test_loader)
    
    models_[i]['Training Loss'] = tt.train_losses
    models_[i]["Training Accuracy"] = tt.train_acc[4000:]
    models_[i]['Test Loss'] = tt.test_losses
    models_[i]["Test Accuracy"] = tt.test_acc

#%%
#import matplotlib.pyplot as plt
#
#fig, axs = plt.subplots(2,2,figsize=(15,10))
#axs[0, 0].plot(tt.train_losses)
#axs[0, 0].set_title("Training Loss")
#axs[1, 0].plot(tt.train_acc[4000:])
#axs[1, 0].set_title("Training Accuracy")
#axs[0, 1].plot(tt.test_losses)
#axs[0, 1].set_title("Test Loss")
#axs[1, 1].plot(tt.test_acc)
#axs[1, 1].set_title("Test Accuracy")
##
