
from __future__ import print_function
import torch
import torch.optim as optim


import os
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Desktop\TSAI\Session 6\Modules')
from GBNModel import GBNNet
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
model = GBNNet().to(device)
summary(model, input_size=(1, 28, 28))
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
#%%
from torch.optim.lr_scheduler import StepLR

models_ = {
            1:{'L1' : False,'L2' : False,'GBN':False},
            2:{'L1' : False,'L2' : False,'GBN':True},
            3:{'L1' : True,'L2' : False,'GBN':False},
            4:{'L1' : True,'L2': False,'GBN':True},
            5:{'L1' : False,'L2' : True,'GBN':False},
            6:{'L1' : False,'L2' : True,'GBN':True},
            7:{'L1' : True,'L2' : True,'GBN':False},
            8:{'L1' : True,'L2': False,'GBN':True}
            }

for i in list(range(1,9)):
    if (models_[i]['GBN']==False):
        model =  Net().to(device)
    else:
        model =  GBNNet().to(device)
    
    def Optimizer_(L2 = models_[i]['L2']):
        if (L2 == False):
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9,weight_decay = 0.00005)
        return optimizer
    optimizer = Optimizer_(L2 = True)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    tt = TrainTest()
    EPOCHS = 25
    test_acc = 0
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        tt.train_(model, device, train_loader, optimizer, epoch,L1 = models_[i]['L1'])
        scheduler.step()
        acc = tt.test_(model, device, test_loader)
        if acc>=test_acc:
            models_[i]['Saved Model'] = model.state_dict()
                
    models_[i]['Training Loss'] = tt.train_losses
    models_[i]["Training Accuracy"] = tt.train_acc[4000:]
    models_[i]['Test Loss'] = tt.test_losses
    models_[i]["Test Accuracy"] = tt.test_acc

#%%
import matplotlib.pyplot as plt
for i in list(range(1,9)):
    plt.plot(models_[i]['Test Accuracy'],label='L1:{},L2:{},GBN:{}'.format(models_[i]['L1'],models_[i]['L2'],models_[i]['GBN']))
    plt.legend()
#%%
for i in list(range(1,9)):
    plt.plot(models_[i]['Test Loss'],label='L1:{},L2:{},GBN:{}'.format(models_[i]['L1'],models_[i]['L2'],models_[i]['GBN']))
    plt.legend()

#%%
import numpy as np
#from google.colab import files

for m_id in [1,2]:
    model = GBNNet().to(device)
    model.load_state_dict(models_[m_id]['Saved Model'])
    model.eval()
    
    correct = 0
    pred_wrong = []
    true_wrong = []
    image = []
    pred_wrong_t=[]
    true_wrong_t=[]
    image_t=[]
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        preds = (pred.cuda()).cpu().numpy()
        target = (target.cuda()).cpu().numpy()
    
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = (data.cuda()).cpu().numpy()
        
        for i in range(len(preds)):
    
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])
                
        pred_wrong_t.extend(pred_wrong)
        true_wrong_t.extend(true_wrong)
        image_t.extend(image)
    
    
    
    figure = plt.figure()
    num_of_images = 25
    for index in range(1, num_of_images + 1):
        plt.subplot(5,5, index)
        plt.subplots_adjust(hspace=.9)
        plt.axis('off')
        plt.imshow(image_t[index].squeeze(), cmap='gray_r')
        a='P-'+str(int(pred_wrong_t[index]))+' A-'+str(int(true_wrong_t[index]))
        plt.title(a)
    
    label='L1-{},L2-{},GBN-{}'.format(models_[m_id]['L1'],models_[m_id]['L2'],models_[m_id]['GBN'])
    figure.savefig(label+'.png')


