#!/usr/bin/python
"""
script for training a cnn on the spectrograms generated of the gravityspy glitch dataset 
using the transfer learning method
"""

import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
import numpy as np
import h5py
from pathlib import Path
import time
import keras
from keras.layers import Flatten, Dense, Dropout, Input, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback
from sklearn.preprocessing import LabelEncoder
import sys
# sys.path.append('/dovilabfs/work/tommaria/gw/tools')
# from load_dataset import *

# extract_model = 'vgg16'
# extract_model = 'inceptionv3'
# extract_model = 'xception'
extract_model = 'resnet152v2'
# extract_model = 'inception_resnetv2'

input_size = (299, 299)

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')

with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_split_wrap_no_ty.hdf5'), 'r') as f:
	x_train = np.asarray(f['x_train'])
	x_test = np.asarray(f['x_test'])
	x_val = np.asarray(f['x_val'])
	y_train = [item.decode('ascii') for item in list(f['y_train'])]
	y_test = [item.decode('ascii') for item in list(f['y_test'])]
	y_val = [item.decode('ascii') for item in list(f['y_val'])]

# encode labels to integers
encoder = LabelEncoder().fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
y_val = encoder.transform(y_val)

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_augmentation3_wrap_no_ty.hdf5'), 'r') as f:
	x_aug = np.asarray(f['x_aug'])
	y_aug = [item.decode('ascii') for item in list(f['y_aug'])]
    
y_aug = encoder.transform(y_aug)
y_aug = keras.utils.to_categorical(y_aug, num_classes)

del encoder

# resize images if required
if x_train.shape[1:3] != input_size:
	x_train = np.asarray([resize(img, input_size, mode='reflect') for img in x_train])
	x_test = np.asarray([resize(img, input_size, mode='reflect') for img in x_test])
	x_val = np.asarray([resize(img, input_size, mode='reflect') for img in x_val])
	x_aug = np.asarray([resize(img, input_size, mode='reflect') for img in x_aug])

x_ta = np.append(x_train, x_aug, axis=0)
y_ta = np.append(y_train, y_aug, axis=0)

input_shape = x_train[0].shape

# Define network

if extract_model == 'vgg16':
	from keras.applications.vgg16 import VGG16
	trained_model = VGG16(include_top=False, input_shape=input_shape)
elif extract_model == 'inceptionv3':
	from keras.applications.inception_v3 import InceptionV3
	trained_model = InceptionV3(include_top=False, input_shape=input_shape)
elif extract_model == 'xception':
	from keras.applications.xception import Xception
	trained_model = Xception(include_top=False, input_shape=input_shape)
elif extract_model == 'resnet152v2':
	from keras.applications.resnet_v2 import ResNet152V2
	trained_model = ResNet152V2(include_top=False, input_shape=input_shape)
elif extract_model == 'inception_resnetv2':
	from keras.applications.inception_resnet_v2 import InceptionResNetV2
	trained_model = InceptionResNetV2(include_top=False, input_shape=input_shape)

output = trained_model.layers[-1].output
output = GlobalMaxPooling2D()(output)
output = Dense(100, activation='relu')(output)
output = Dense(num_classes, activation='softmax')(output)
model = Model(input=trained_model.input, output=output)
model.summary()

train_set = 'fromraw_gs_wrap_no_ty'
opt_method = 'adadelta'
model_num = '15'
model_name = 'gpu_test_' + model_num
model_path = Path('/dovilabfs/work/tommaria/gw/gravityspy/gpu/' + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name)
weights_file = join(model_path, model_name + '.weights.best.hdf5')
if not os.path.exists(model_path):
    os.makedirs(model_path)

print('Model: ' + opt_method + '/' + extract_model + '/' + model_name + '\n')

# Compile the model

optimizer = Adagrad(lr=1e-4, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_method, metrics=['accuracy'])

# Train the model

checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, 
                               save_best_only=True, monitor='val_accuracy')#'val_loss')
progbar = TQDMCallback()

batch_size = 32
epochs = 20

start_time = time.time()

model.fit(x_ta, y_ta, batch_size=batch_size, epochs=epochs, 
          callbacks=[checkpointer, progbar], validation_data=(x_val, y_val), verbose=2, shuffle=True)

print("--- Network training time is %.7s minutes ---\n" % ((time.time() - start_time) / 60.0))

model.save(join(model_path, model_name + '.h5'))

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
