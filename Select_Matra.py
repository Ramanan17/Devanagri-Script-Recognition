# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:40:58 2019

@author: Ramanan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:56:54 2019

@author: Ramanan
"""

#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook

from sklearn.utils import shuffle
import cv2
#from resnets_utils import *

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#%%
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
#%%
filelist=os.listdir('..\Layer')
Training_images  =[] 
labels=[]
labels2 =[]
i=0
for const in filelist:
    pics = os.listdir('..\Layer\{}'.format(const))
    i=i+1 
    for pic in pics:
        image = Image.open('..\Layer\{}\{}'.format(const,pic))
        image = image.resize((64,64), Image.ANTIALIAS)
       
        image1 = np.array(image)
        labels.append([const,1])
        labels2.append(const)
        Training_images.append(image1)    
       

         
        
#%%
Y = np.array(pd.get_dummies(labels2))        

X_train_orig, X_test_orig,  Y_train,Y_test = train_test_split(Training_images, Y)
 


#%%
X_train= np.array(X_train_orig)/255
X_test = np.array(X_test_orig)/255


#%%
img_height,img_width = 64,64 
num_classes = 50
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
#%%
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
#%%
from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])



#%%
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

#%%
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=100)


#%%



#for heavy model architectures, .h5 file is unsupported.
weigh= model.get_weights()
#%%
with open('Select_Matrra', 'wb') as fp:
     pickle.dump(weigh, fp)
with open ('Select_Matrra', 'rb') as fp:
     weighlist = pickle.load(fp)
     
#%%
import pickle     
with open ('Select_Matrra', 'rb') as fp:
     weighlist = pickle.load(fp)
#%%
model.set_weights(weighlist)         
#%%
preds = model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))    
