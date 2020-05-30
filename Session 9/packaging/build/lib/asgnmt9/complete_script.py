from torchvision import transforms
from DataLoader import cifar_data_loader,View_images
transform_params = transforms.Compose([transforms.RandomHorizontalFlip(),  
                                       transforms.RandomRotation(10),  
                                       transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),  
                                       transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader,testloader = cifar_data_loader(transform_params,BatchSize=4)
#View_images(trainloader,classes)

import torch

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

from models.resnet import ResNet18
net = ResNet18().to(device)

summary(net, input_size=(3,32,32))

from Training_Testing import TrainTest
import torch.optim as optim
import torch.nn as nn

tt = TrainTest()
test_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(8):
    print("EPOCH:", epoch)
    tt.train_(net, device, trainloader, optimizer, criterion, epoch,L1 = False)
#    scheduler.step()
    acc = tt.test_(net, device, testloader)  
    
   
print('Finished Training')