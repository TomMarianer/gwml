#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:39:36 2019

@author: tommarianer

script for finding gravityspy glitches which are missing from our dataset 
(the files containing them were not downloaded because they were not publicly available)
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
idx = []

start_time = time.time()
for dfidx in tqdm(dfidxs):
	t_i = (metadata['event_time'][dfidx]) - Tc/2
	t_f = (metadata['event_time'][dfidx]) + Tc/2
	detector = metadata['ifo'][dfidx][0]
	files = get_files(detector)
	chunks = get_chunks(t_i, t_f, Tc, To)
	for chunk in chunks:
		try:
			data = TimeSeries.read(files, start=chunk[0], end=chunk[1], format='hdf5.losc') # load data locally
		except:
			idx.append(dfidx)
			break

print(len(idx))
missing = metadata.iloc[idx, :]
missing.to_csv('missing.csv')
print("--- Execution time is %.7s seconds ---\n" % ((time.time() - start_time)))
