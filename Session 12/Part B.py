# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
# import numpy as np
import math
import json
import os
dogs_json = open('/home/jai/Documents/TSAI-EVA40-Assignments/Session 12/Assignemnt 12.json')
dogs_json = json.load(dogs_json)

req = dogs_json['_via_img_metadata']
req = pd.DataFrame(req.items())
dogs_oths = pd.DataFrame(columns = ['Image Name',
                                    'Shape Name',
                                    'BB x','BB y',
                                    'BB width',
                                    'BB height',
                                    'Class'])

n =0 
i,j = 0,0
for i in range(0,req.shape[0]):
    for j in range(0,len(req.iloc[i,1]['regions'])):
        dogs_oths.loc[n,'Image Name'] = req.iloc[i,1]['filename']
        dogs_oths.loc[n,'Shape Name'] = req.iloc[i,1]['regions'][j]['shape_attributes']['name']
        dogs_oths.loc[n,'BB x'] = req.iloc[i,1]['regions'][j]['shape_attributes']['x']
        dogs_oths.loc[n,'BB y'] = req.iloc[i,1]['regions'][j]['shape_attributes']['y']
        dogs_oths.loc[n,'BB width'] = req.iloc[i,1]['regions'][j]['shape_attributes']['width']
        dogs_oths.loc[n,'BB height'] = req.iloc[i,1]['regions'][j]['shape_attributes']['height']
        dogs_oths.loc[n,'Class'] = req.iloc[i,1]['regions'][j]['region_attributes']['Class']
        n +=1
        # print(n)
    
#%%
os.chdir(r'/home/jai/Documents/TSAI-EVA40-Assignments/Session 12/Dogs')
os.listdir()

data = pd.DataFrame({'Image Name':os.listdir()})
data['Image width'] = 0
data['Image height'] = 0
import PIL

for i in os.listdir():
    image = PIL.Image.open(i)
    # print(image.size)
    sers = (data['Image Name'] == i)
    data.loc[sers,'Image width'], data.loc[sers,'Image height'] = image.size
    
dogs_oths = pd.merge(dogs_oths,data, how = 'left',on = 'Image Name')

#%%
## Finding Centroids
dogs_oths['BBx Cx'] = dogs_oths['BB x'] + (dogs_oths['BB width']/2)
dogs_oths['BBy Cy'] = dogs_oths['BB y'] + (dogs_oths['BB height']/2)

#%%
final = dogs_oths.copy()
final['N width'] = final['Image width']/final['Image width']
final['N height'] = final['Image height']/final['Image height']

final['N BB width'] = final['BB width']/final['Image width']
final['N BB height'] = final['BB height']/final['Image height']

final['N BBx Cx'] = (final['BBx Cx']/(final['Image width']/13)).apply(lambda x:math.ceil(x))
final['N BBy Cy'] = (final['BBy Cy']/(final['Image height']/13)).apply(lambda x:math.ceil(x))
