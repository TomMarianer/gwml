#!/usr/bin/python
"""
script for splitting the dataset into training test and validation sets
as well as augmenting the training set
"""

import keras
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, tqdm_notebook
from skimage import io
from skimage.transform import resize
import numpy as np
from pathlib import Path
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import h5py

from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.util import random_noise
import random

def randRange(a, b):
	'''
	a utility functio to generate random float values in desired range
	'''
	return random.random() * (b - a) + a

def randomAffine(im):
	'''
	wrapper of Affine transformation with random scale, rotation, shear and translation parameters
	'''
	tform = AffineTransform(translation=(randRange(-int(0.5 * im.shape[0]), int(0.5 * im.shape[0])), 0))
	return warp(im, tform.inverse, mode='wrap')

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')

with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs.hdf5'), 'r') as f:
	x = np.asarray(f['x'])
	y = [item.decode('ascii') for item in list(f['y'])]
	times = list(f['times'])

temp = []

for jdx, val in enumerate(y):
	temp.append([val, times[jdx]])

y = np.asarray(temp)

x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []

print(x.shape)

for label in np.unique(y[:,0]):
	idx = y[:,0] == label
	x_temp, x_test_temp, y_temp, y_test_temp = train_test_split(x[idx], y[idx], test_size=0.2, random_state=0)
	if x_temp.shape[0] == 1:
		x_train_temp = x_temp
		y_train_temp = y_temp
		x_val_temp = []
		y_val_temp = []

	else:
		x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(x_temp, y_temp, test_size=0.1, random_state=0)

	x_train.extend(x_train_temp)
	y_train.extend(y_train_temp)
	x_test.extend(x_test_temp)
	y_test.extend(y_test_temp)
	x_val.extend(x_val_temp)
	y_val.extend(y_val_temp)
	
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_val = np.asarray(x_val)

y_train = np.asarray(y_train)
times_train = y_train[:,1].astype('float64')
y_train = y_train[:,0]

y_test = np.asarray(y_test)
times_test = y_test[:,1].astype('float64')
y_test = y_test[:,0]

y_val = np.asarray(y_val)
times_val = y_val[:,1].astype('float64')
y_val = y_val[:,0]


data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')
with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_split_wrap_no_ty.hdf5'), 'w') as f:
	f.create_dataset('x_train', data=x_train)
	f.create_dataset('x_test', data=x_test)
	f.create_dataset('x_val', data=x_val)
	f.create_dataset('y_train', data=[item.encode('ascii') for item in y_train])
	f.create_dataset('y_test', data=[item.encode('ascii') for item in y_test])
	f.create_dataset('y_val', data=[item.encode('ascii') for item in y_val])
	f.create_dataset('times_train', data=times_train)
	f.create_dataset('times_test', data=times_test)
	f.create_dataset('times_val', data=times_val)

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(len(y_train))
print(len(y_test))
print(times_train.shape)

del x_test, y_test, times_train, times_test

labels = np.unique(y_train)

counts = {}
for label in labels:
    counts[label] = list(y_train).count(label)

x_aug = []
y_aug = []

for idx, img in enumerate(x_train):
    x_aug.append(randomAffine(img))
    y_aug.append(y_train[idx])
    
    if counts[y_train[idx]] < 1000:
        for tmp in range(3):
            x_aug.append(randomAffine(img))
            y_aug.append(y_train[idx])
            
    if counts[y_train[idx]] < 200:
        for tmp in range(6):
            x_aug.append(randomAffine(img))
            y_aug.append(y_train[idx])

x_aug = np.asarray(x_aug)

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')
with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_augmentation3_wrap_no_ty.hdf5'), 'w') as f:
	f.create_dataset('x_aug', data=x_aug)
	f.create_dataset('y_aug', data=[item.encode('ascii') for item in y_aug])

print(x_aug.shape)
