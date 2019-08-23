# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:24:38 2019

@author: Ramanan
"""

#%%
import numpy as np
import pandas as  pd
import os
from PIL import Image
from PIL import ImageOps
#%%
filelist=os.listdir('F:\cvpr\matras\Bottom')

i=0
for matra in filelist:
     imgs = os.listdir('F:\cvpr\matras\Bottom\{}'.format(matra))
     for img in imgs:
         to_crop = Image.open('F:\cvpr\matras\Bottom\{}\{}'.format(matra,img)).convert('L')
         to_crop = to_crop.resize((32,32),Image.ANTIALIAS)
         n_array = np.array(to_crop)
         im_final = Image.fromarray(n_array[20:,:])
         im_final.save('F:\cvpr\matras\Bottom\{}\{}'.format(matra,img))
         
         