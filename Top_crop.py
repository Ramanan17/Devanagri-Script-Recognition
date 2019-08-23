# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:53:27 2019

@author: Ramanan
"""

#%%
import numpy as np
import pandas as  pd
import os
from PIL import Image
from PIL import ImageOps
#%%
filelist=os.listdir('F:\cvpr\matras_nissy\matras\no_matras')

i=0
for matra in filelist:
     imgs = os.listdir('F:\cvpr\matras_nissy\matras\no matras\{}'.format(matra))
     for img in imgs:
         to_crop = Image.open('F:\cvpr\matras_nissy\matras\no matras\{}\{}'.format(matra,img)).convert('L')
         to_crop = to_crop.resize((32,32),Image.ANTIALIAS)
         n_array = np.array(to_crop)
         im_final = Image.fromarray(n_array[:16,:])
         im_final.save('F:\cvpr\matras_nissy\matras\no_matras_top\{}\{}'.format(matra,img))
         
         
         