#!/usr/bin/python3

import numpy as np
import time
from os import listdir, makedirs
from os.path import isfile, join, exists
from pathlib import Path
import h5py

# inj_type = 'sg'
# inj_type = 'rd'
# inj_type = 'ga'
# inj_type = 'cg'
# inj_type = 'cg_inc'
inj_type = 'cg_double'
# inj_type = 'wn'

detector = 'L'
data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/injected/tomove')

start_time = time.time()

files = [f for f in sorted(listdir(data_path)) if (isfile(join(data_path, f)) and f.endswith('.hdf5'))]

x = []
times = []

for file in files:
	# print(file)
	with h5py.File(join(data_path, file), 'r') as f:
		x.extend(np.asarray(f['x']))
		times.extend(np.asarray(f['times']))

x = np.asarray(x)
times = np.asarray(times)

print(x.shape)
print(times.shape)

data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/injected/tomove/combined')
if not exists(data_path):
	makedirs(data_path)

if inj_type == 'cg_inc' or inj_type == 'cg_double':
	fname = '-'.join(files[0].split('-')[:-2]) + '.hdf5'

print(fname)

with h5py.File(join(data_path, fname), 'w') as f:
	f.create_dataset('x', data=x)
	f.create_dataset('times', data=times)

print(detector)
print("--- Execution time is %.7s seconds ---\n" % (time.time() - start_time))

# num = 7
# if inj_type == 'cg_inc':
# 	params = pd.read_csv('/storage/home/tommaria/thesis/tools/' + inj_type + '_params_csv.csv', usecols=['f0', 'Q', 'A'])
# 	f0 = params['f0'][num]
# 	Q = params['Q'][num]
# 	A = params['A'][num]
