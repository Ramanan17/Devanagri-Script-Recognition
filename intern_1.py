# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:41:22 2019

@author: Ramanan
"""



#%%
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2gray

images_name=os.listdir('consonants')

i=0
hindi_words = []

for name in images_name:
    for src in os.listdir('consonants/{}'.format(name)):
        src = 'consonants/{}/{}'.format(name,src)
        image = Image.open(src,'r')
        image = image.resize((64,64),Image.ANTIALIAS)
        image1 = np.array(image)[...,:3]
      
        hindi_words.append(image1)
        
hindi_words = np.array(hindi_words)        
        
      
 

#%%
Digits =[]
digits = os.listdir('Digits')
for src in digits:
    src='Digits/{}'.format(src)
    image = Image.open(src,'r')
    #image = image.resize((64,64))
    image = image.resize((64,64),Image.ANTIALIAS)
    image1 = np.array(image)[...,:3]
    Digits.append(image1)
Digits = np.array(Digits)
#%%
English_letters =[]
words = os.listdir('Letters')
for src in words:
    src ='Letters/{}'.format(src)
    image = Image.open(src,'r')
    image = image.resize((64,64),Image.ANTIALIAS)
    image1 = np.array(image)[...,:3]
    English_letters.append(image1)
English_letters = np.array(English_letters)
#%%
X=np.zeros([8515,64,64,3])
X[:5105] = hindi_words 
X[5106:5656] = Digits
X[5655:] = English_letters

#%%
y =np.ones([8515])
y[:5105] = 1   ## 1- hindi-word
y[5106:5656] = 2 ## 2-Digits
y[5655:] = 3 ##3-Letters
 
#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y) 
#le.fit([1,2,3])   
#%%
 
#%%   
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#%%
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dense, BatchNormalization,Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D,Activation
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import SGD, Adam
#%%
from keras import backend as k

#%%
def get_resnet_model():
  img_height,img_width = 64,64 
  num_classes = 3
  base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (3,img_height,img_width))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.7)(x)
  predictions = Dense(num_classes, activation= 'softmax')(x)
  model = Model(inputs = base_model.input, outputs = predictions)
 
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
  adam = Adam(lr=0.0001)
  model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
 ## model.load_weights('G:\\cvpr\\mser\\final_model_fold3_weights.h5')
  return model



#%%
import keras
from keras.preprocessing.image import ImageDataGenerator
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#%%

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
#%%
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]
#%%
y_train =np_utils.to_categorical(y_train,3)
y_test = np_utils.to_categorical(y_test,3)
#%%

datagen.fit(np.rollaxis(X_train, 3, 1))
#%%

#%%
X_train = np.rollaxis(X_train, 3, 1)
X_test =np.rollaxis(X_test, 3, 1)
#%%
model  = get_resnet_model()
    # Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=64),
                        epochs=100,
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) / 64)
    
model.save_weights('intern_task_1_weights.h5')
model.save_weights('intern_task_1_weights_with_model.h5')
#%%
score, acc = model.evaluate(X_test, y_test,
                            batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
#%%
test = np.zeros([1,64,64])
test[0] =Digits[21]
predict = np.zeros([1,64,64,1])
#predict[0] = X_train[5106:5656]
out =model.predict_classes(X_test)
#%%
from predict import predict
predict()
