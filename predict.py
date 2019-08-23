# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:00:25 2019

@author: VR LAB PC3
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
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
from PIL import ImageOps
from keras.optimizers import SGD, Adam
#%%
def load_image(src):
    image = Image.open(src)
    image = image.resize((64,64), Image.ANTIALIAS)
    resnet_image = (np.array(image.getdata()).reshape(image.size[0], image.size[1], 3))/255
    to_crop = image.resize((64,64),Image.ANTIALIAS)
    n_array = np.array(to_crop)
    top_image = Image.fromarray(n_array[:15,:])
    top_image = top_image.resize((64,64),Image.ANTIALIAS)
    top_image =(np.array(top_image.getdata()).reshape(image.size[0], image.size[1], 3))/255
    to_crop = image.resize((32,32),Image.ANTIALIAS)
    n_array = np.array(to_crop)
    bottom_image = Image.fromarray(n_array[25:,:])
    bottom_image = bottom_image.resize((64,64),Image.ANTIALIAS)
    bottom_image =(np.array(bottom_image.getdata()).reshape(image.size[0], image.size[1], 3))/255
    vertical_image = Image.fromarray(n_array[:,19:])
    vertical_image = vertical_image.resize((64,64),Image.ANTIALIAS)
    vertical_image =(np.array(vertical_image.getdata()).reshape(image.size[0], image.size[1], 3))/255
    return [resnet_image,top_image,bottom_image,vertical_image]


#%%    
def get_resnet_model():
  img_height,img_width = 64,64 
  num_classes = 50
  base_model = applications.resnet50.ResNet50(weights= None, include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.7)(x)
  predictions = Dense(num_classes, activation= 'softmax')(x)
  model = Model(inputs = base_model.input, outputs = predictions)
 
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
  adam = Adam(lr=0.0001)
  model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
  model.load_weights('G:\\cvpr\\mser\\final_model_fold3_weights.h5')
  return model

  
#%%  
def top_model():
    model = Sequential()
    num_classes =6
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64,64,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu'))
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('G:\\cvpr\\matras\\Top_weights\\Top_model_fold0_weights.h5')
    return model
#%%
def bottom_model():
    model = Sequential()
    num_classes =3
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64,64,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu'))
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('G:\\cvpr\matras\Bottom_weights\\Bottom_model_fold2_weights.h5')
    return model
#%%
def vertical_model():
    model = Sequential()
    num_classes=5
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64,64,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu'))
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('G:\\cvpr\\matras\\Vertical_weights\\Vertical_model_fold7_weights.h5')
    return model    

#%%
resnet_model = get_resnet_model()
top = top_model()
Bottom = bottom_model()
vertical = vertical_model()
#%% Label encoder
filelist_resent=os.listdir('F:\\cvpr\\consonants')    
from sklearn import preprocessing
resnet_label = preprocessing.LabelEncoder()
resnet_label.fit(filelist_resent)
#%%
filelist_top=os.listdir('F:\\cvpr\\matras\\Top')   

top_label = preprocessing.LabelEncoder()
top_label.fit(filelist_top)
#%%
filelist_bottom=os.listdir('F:\\cvpr\\matras\\Bottom')    
#filelist_temp = filelist_bottom[10:]
#filelist_temp.append(filelist_top[10]) 
from sklearn import preprocessing
bottom_label = preprocessing.LabelEncoder()
bottom_label.fit(filelist_bottom)
#%%
filelist_vertical=os.listdir('F:\\cvpr\\matras\\Vertical')   
 
from sklearn import preprocessing
vertical_label = preprocessing.LabelEncoder()
vertical_label.fit(filelist_vertical)  
#%%
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend as K
img = load_image('F:\\cvpr\\Ha.png')
x = image.img_to_array(img[0])
arr = np.zeros([1,64,64,3])
arr[0] =x
import matplotlib.pyplot as plt
plt.imshow(img[0])

preds = resnet_model.predict(arr)
index = np.argmax(preds)
word = resnet_label.inverse_transform([index])
print(word)
top_image =image.img_to_array(img[1])
y = image.img_to_array(img[1])
arr_top = np.zeros([1,64,64,3])
arr_top[0] =y
preds_top = top.predict(arr_top)
index = np.argmax(preds_top)
word = top_label.inverse_transform([index])
print(word)

bottom_image =image.img_to_array(img[2])
y = image.img_to_array(img[2])
arr_top = np.zeros([1,64,64,3])
arr_top[0] =y
preds_bottom = Bottom.predict(arr_top)
index = np.argmax(preds_bottom)
word = bottom_label.inverse_transform([index])
print(word)
vertical_image =image.img_to_array(img[3])
y = image.img_to_array(img[3])
arr_top = np.zeros([1,64,64,3])
arr_top[0] =y
preds_vertical = vertical.predict(arr_top)
index = np.argmax(preds_vertical)
word = vertical_label.inverse_transform([index])
print(word)

    