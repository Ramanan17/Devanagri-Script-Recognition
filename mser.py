# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:54:56 2019

@author: Ramanan
"""

#%%
import cv2
import numpy as np
import hashlib
#from scipy.misc import imread,imshow,imresize
from hashlib import md5
from skimage.transform import resize
mser = cv2.MSER_create()


#replace it with your file location
img = cv2.imread("sample.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
height = 2**6
width = 2**6
#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)


hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)


i=0
finalimages =[]
for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    x, y, width, height = cv2.boundingRect(contour)
    roi = img[y:y+height, x:x+width]
    if(width<25 and height<25):
        continue
    for x in range(0,width,38):
        if(x+40<=width):
            write = roi[:,x:x+48]
           # cv2.imwrite("out"+str(i)+".png", write)
            
        
        
    #imgwidth, imgheight = roi.size
    
            
    
    
    
    finalimages.append(roi)
    i=i+1

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)
storeimages = finalimages

   
   

cv2.imshow("text only", text_only)

cv2.waitKey(0)

#%%
