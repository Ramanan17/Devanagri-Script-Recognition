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
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#%%
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
#%%
filelist=os.listdir('.\consonants')    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(filelist)    

#%%
    

Training_images  =[] 
labels=[]
labels2 =[]
i=0
for const in filelist:
    pics = os.listdir('.\consonants\{}'.format(const))
    i=i+1 
    for pic in pics:
        image = Image.open('.\consonants\{}\{}'.format(const,pic))
        image = image.resize((64,64), Image.ANTIALIAS)
        image1 = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
        labels.append([const,1])
        labels2.append(const)
        Training_images.append(image1)    
       

         
        
#%%
Y =le.transform(labels2)   

X_train_orig, X_test_orig,  Y_train_orig,Y_test_orig = train_test_split(Training_images, Y)
 


#%%
X_train= np.array(X_train_orig)/255
X_test = np.array(X_test_orig)/255
Y_train = np_utils.to_categorical(Y_train_orig,50)
Y_test = np_utils.to_categorical(Y_test_orig,50)

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
#%%
def create_model():
    img_height,img_width = 64,64 
    num_classes = 50
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))

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
def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]
#%%
batch_size=32

gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10
                        )    
#%%    
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=1).split(X_train, Y_train))
cvscores = []
for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = Y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv= Y_train[val_idx]
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
    generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
    
    model.fit_generator(
                generator,
                steps_per_epoch=len(X_train_cv)/batch_size,
                epochs=15,
                shuffle=True,
                verbose=1,
                validation_data = (X_valid_cv, y_valid_cv),
                callbacks = callbacks)
    
    
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=100)

print(model.evaluate(X_valid_cv, y_valid_cv))
#%%



#for heavy model architectures, .h5 file is unsupported.

#%%
model.set_weights(weighlist)         
#%%
preds = model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))    
