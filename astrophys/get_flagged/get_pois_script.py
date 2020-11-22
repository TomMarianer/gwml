#!/usr/bin/env python3
"""
scirpt for extracting spectrograms of the time-stamps flagged as outliers and saving them to a single file
"""

import git
from os import listdir
from os.path import isfile, join, dirname, realpath

def get_git_root(path):
	"""Get git root path
	"""
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root

file_path = dirname(realpath(__file__))
git_path = get_git_root(file_path)

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append(git_path + '/astrophys/tools')
from tools_gs_par import *

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
detector = 'H'
print(detector)

# method = 'kde'
# method = 'gram_single'
# method = 'y_hat_chirp'
# method = 'y_hat_nota'

method = 'hardware_inj'

with h5py.File(git_path + '/shared/tois.hdf5', 'r') as f:
	tois = np.asarray(f[method])

print(len(tois))

# data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
data_path = Path('/arch/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
files = [join(data_path, f) for f in sorted(listdir(data_path)) if isfile(join(data_path, f)) and f.split('.')[-1] == 'hdf5']

x = []
times = []

for t in tois:
	file = find_file(t, files)
	if file is None:
		continue

	with h5py.File(join(data_path, file), 'r') as f:
		temp_times = np.asarray(f['times'])

	if method == 'hardware_inj':
		idx = find_closest_index(t - 0.5, temp_times)
		if idx is None:
			continue

	else:
		idx = find_closest_index(t + 0.5, temp_times)

	with h5py.File(join(data_path,file), 'r') as f:
		x.append(np.asarray(f['x'][idx]))

	times.append(temp_times[idx])

x = np.asarray(x)
times = np.asarray(times)

# data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/moved')
data_path = Path('/arch/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined/pois')
if not exists(data_path):
	makedirs(data_path)

pois_file = 'pois_new_gram_' + method + '.hdf5'

with h5py.File(join(data_path, pois_file), 'w') as f:
	f.create_dataset('x', data=x)
	f.create_dataset('times', data=times)

print(x.shape)
print(times[21])
print(times.shape)

print(tois - times)

tot_time = 0
for file in files:
	with h5py.File(join(data_path, file), 'r') as f:
		tot_time += len(np.asarray(f['times']))

print(tot_time)

print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
