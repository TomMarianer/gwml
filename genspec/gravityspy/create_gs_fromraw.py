#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:39:36 2019

@author: tommarianer

script for generating spectrograms of the gravityspy glitch dataset from raw strain data
"""

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
from tools_gs import *
from params import *
import pandas as pd

metadata = pd.read_csv('trainingset_v1d1_metadata.csv', 
					   usecols=['event_time', 'ifo', 'label', 'gravityspy_id'])

dfidxs = metadata.index
x = []
y = []
times = []
idx_start = 7000
idx_stop  = np.min([len(dfidxs), idx_start + 1000])
for dfidx in tqdm(dfidxs[idx_start:idx_stop]):
	t_i = (metadata['event_time'][dfidx]) - Tc/2
	t_f = (metadata['event_time'][dfidx]) + Tc/2
	detector = metadata['ifo'][dfidx][0]
	temp = condition_chunks(t_i, t_f, Tc=Tc, To=To, local=True, fw=fw, qtrans=True, 
	qsplit=False, dT=dT,
	detector=detector, save=False)
	if temp is None:
		continue

	t0 = []
	for img in temp:
		t0.append(img.t0)

	tdiff = t0 - metadata['event_time'][dfidx]
	idx = np.argmax(tdiff[tdiff < 0])
	x.append(temp[idx].values)
	y.append(metadata['label'][dfidx])
	times.append(metadata['event_time'][dfidx])

x = np.asarray(x)
print(x.shape)

import h5py

y = [item.encode('ascii') for item in y]

data_path = Path('/storage/fast/users/tommaria/data/gravityspy/fromraw/training')

with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_' + str(idx_start).zfill(4) + '_' + str(idx_stop).zfill(4) + '_gs.hdf5'), 'w') as f:
	f.create_dataset('x', data=x)
	f.create_dataset('y', data=y)
	f.create_dataset('times', data=times)
