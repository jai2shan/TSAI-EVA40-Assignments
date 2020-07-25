
import numpy as np
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
import pickle
import math
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet101
import natsort
os.chdir(r'D:\Assignment 15A\Depth-Estimation-PyTorch')

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class I2D(nn.Module):
    def __init__(self, pretrained=True):
        super(I2D, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1) # 256
        self.layer2 = nn.Sequential(resnet.layer2) # 512
        self.layer3 = nn.Sequential(resnet.layer3) # 1024
        self.layer4 = nn.Sequential(resnet.layer4) # 2048

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Depth prediction
        self.predict1 = smooth(256, 64)
        self.predict2 = predict(64, 1)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size() # batchsize N,channel,height,width
        
        # Bottom-up
        c1 = self.layer0(x) 
        c2 = self.layer1(c1) # 256 channels, 1/4 size
        c3 = self.layer2(c2) # 512 channels, 1/8 size
        c4 = self.layer3(c3) # 1024 channels, 1/16 size
        c5 = self.layer4(c4) # 2048 channels, 1/32 size

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # 256 channels, 1/16 size
        p4 = self.smooth1(p4) 
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # 256 channels, 1/8 size
        p3 = self.smooth2(p3) # 256 channels, 1/8 size
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # 256, 1/4 size
        p2 = self.smooth3(p2) # 256 channels, 1/4 size

        return self.predict2( self.predict1(p2) )     # depth; 1/4 size, mode = "L"

class NYUv2Dataset(data.Dataset):
    def __init__(self):
        self.name_map = pickle.load(open("./nyuv2/index.pkl",'rb'))
        self.rgb_paths = list(self.name_map.keys())
        self.rgb_transform = Compose([ToTensor()])
        self.depth_transform = Compose([ToTensor()])
        self.length = len(self.rgb_paths)
            
    def __getitem__(self, index):
        path = './nyuv2/test_rgb/'+self.rgb_paths[index]
        rgb = Image.open(path)
        depth = Image.open('./nyuv2/test_depth/'+self.name_map[self.rgb_paths[index]])
#         depth.save("gt_origin.ppm")
        depth = depth.resize((160,120))
        return self.rgb_transform(rgb).float(), self.depth_transform(depth).float()

    def __len__(self):
        return self.length
    
class CustomDataSet():
    def __init__(self):
        self.main_dir = r'D:\Assignment 15A\Resized Images\04. Overlaid'
        all_imgs = os.listdir(self.main_dir)
        self.transform = transforms.Compose([transforms.Resize((448,448)),transforms.ToTensor(),])
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
#%%


# dataset
LOAD_DIR = '.'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_dataset = CustomDataSet()
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=25)
i2d = I2D().to(DEVICE)
i2d.load_state_dict(torch.load('{}/fyn_model.pt'.format(LOAD_DIR),map_location='cpu'))
print("model loaded")
# setting to eval mode
i2d.eval()
print('evaluating...')
j=0
m=0
imgs = eval_dataset.total_imgs
with torch.no_grad():
    for i,(data) in enumerate(eval_dataloader):
        print(j)
        j=j+1
        data = data.to(DEVICE)
        pred_depth = i2d(data)
        for i in pred_depth.cpu():
            depth_img = transforms.ToPILImage()(i.int())
            depth_img.save(r'D:\\New Depths'+'\\'+imgs[m].split('.')[0]+".png")
            m=m+1
        # depth_img = transforms.ToPILImage()(pred_depth[1].int().cpu())
        # depth_img.save("pred.png")
        # break


#%%

rgb_resize

#%%

depth_img
#%%

gt_img

#%%


i2d


# ## Test PyTorch: Loading Image ToTensor
# 
# https://blog.csdn.net/Turbo_Come/article/details/99705169

#%%

import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torchvision import transforms
import torch
import random
import matplotlib.pyplot as plt

test_rgb_list = os.listdir("./nyuv2/test_depth")
file = random.choice(test_rgb_list)
img = Image.open("./nyuv2/test_depth/{}".format(file))
W,H = img.size
print("original image from PIL: [WxH] = {}x{}".format(W,H))
print("Type: {}".format(img))
# convert PIL Image to tensor
img_tensor = Compose([ToTensor()])(img).float()
Ch,H,W = img_tensor.shape
print(img_tensor)
print("convert Image to Tensor: [Channel,Height,Width] = {}x{}x{}".format(Ch,H,W))
print("Type of the image tensor: {}".format(type(img_tensor)))
# convert tensor to PIL Image
recon_img = transforms.ToPILImage()(img_tensor.int().squeeze(0))
print("Image recovered from tensor: {}".format(recon_img.size))
print("Type: {}".format(recon_img))
plt.imshow(img,cmap='gray')
# img.save("1.ppm")
#%%


plt.imshow(recon_img,cmap='gray')
recon_img.save("2.ppm")

#%%



