# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:48:08 2019

@author: Ramanan
"""

#%%

#Program to remove duplicates
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread,imshow,imresize
from skimage.transform import resize
import cv2
import os
height = 2**6
width = 2**6
filelist=os.listdir('F:\cvpr\mser')
templist = filelist
for file in filelist:
    if not(file.endswith(".png")):
        continue
    else:
        if(os.path.isfile(file)):
             img1 = imread(file, flatten=True).astype(np.uint8)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
             templist.remove(file)
             for file2 in templist:
                if not(file2.endswith(".png")):
                      continue
                else:
                     if(os.path.isfile(file2)):
                         
                         
                         img2 =  imread(file2, flatten=True).astype(np.uint8)
                        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                         img1 = resize(img1, (height, width))
                         img2 = resize(img2, (height, width))
                         sim, diff = ssim(img1,img2, full=True)
                         if(sim>0.80):
                             os.remove(file2)
                          
                             
                             
                             
                          #  print("Similarity found in {} and {}".format(file,file2))
                             #filelist.remove(file2)
                             
                             
                             
                    
                    
                     
    
                
                     
                     
            
      
            
       
           
        

