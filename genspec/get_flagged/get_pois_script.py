#!/usr/bin/env python3
"""
scirpt for extracting spectrograms of the time-stamps flagged as outliers and saving them to a single file
"""

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
from tools_gs import *
import get_tois

def find_file(t, files):
	"""Return file string containing time t.
	"""

	segments = []
	for file in files:
		t_i = int(file.split('.')[0].split('-')[1])
		t_f = int(file.split('.')[0].split('-')[2])
		segments.append((t_i, t_f))
	# print(segments)

	for segment, file in zip(segments, files):
		if segment[0] <= t and t <= segment[1]:
			return file
	return None

start_time = time.time()

segment_list = get_segment_list('BOTH')
detector = 'L'
print(detector)

# method = 'ooc'
method = 'kde'
# method = 'kde_latent'
# method = 'gram_single'
# method = 'score_alpha_500'
# method = 'y_hat_chirp'
# method = 'y_hat_nota'

tois = get_tois.get_tois(method)

print(len(tois))

# data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
data_path = Path('/arch/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
files = [join(data_path, f) for f in sorted(listdir(data_path)) if isfile(join(data_path, f)) and f.split('.')[-1] == 'hdf5']

x = []
times = []

for t in tois:
	file = find_file(t, files)
	assert(file is not None)
	with h5py.File(join(data_path,file), 'r') as f:
		temp_times = np.asarray(f['times'])

	idx = find_closest_index(t + 0.5, temp_times)
	with h5py.File(join(data_path,file), 'r') as f:
		x.append(np.asarray(f['x'][idx]))

	times.append(temp_times[idx])

x = np.asarray(x)
times = np.asarray(times)

data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
pois_file = '../pois/pois_new_gram_' + method + '.hdf5'

with h5py.File(join(data_path, pois_file), 'w') as f:
	f.create_dataset('x', data=x)
	f.create_dataset('times', data=times)

print(x.shape)
print(times.shape)

print(tois - times)

tot_time = 0
for file in files:
	with h5py.File(join(data_path, file), 'r') as f:
		tot_time += len(np.asarray(f['times']))

print(tot_time)

print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
