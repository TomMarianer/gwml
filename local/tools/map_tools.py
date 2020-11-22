#!/usr/bin/python
"""
tools for handling spectrograms and features for plotting
also includes create_maps function which maps latent space features to map space features
also includes several miscellaneous tools, and tools for unused outlier detection methods
"""

import h5py
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
from umap import UMAP
from pathlib import Path
import joblib
from sklearn.covariance import EmpiricalCovariance

def load_trainingset(condition_method=None, data_path=Path('/Users/tommarianer/LOSC Data/gravityspy/trainingsets'), filename=None):
	"""Load training set for mapping.
	"""

	assert condition_method is not None or filename is not None

	if filename is None:
		filename = 'trainingset_fromraw_centered_2048_Tc_64_' + condition_method + '_split.hdf5'
	with h5py.File(join(data_path, filename), 'r') as f:
		x_train = np.asarray(f['x_train'])
		y_train = [item.decode('ascii') for item in np.asarray(f['y_train'])]
		times_train = np.asarray(f['times_train'])

	return x_train, y_train, times_train

def load_train_features(condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load training features for mapping.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model)
	features_file = 'fromraw_' + condition_method + '_' + '_'.join(model.split('/')) + '.hdf5'
	with h5py.File(join(data_path, features_file), 'r') as f:
		features_train = np.asarray(f['features_train'])
		y_train_hat = np.asarray(f['y_train_hat'])
	return features_train, y_train_hat

def load_labelled_set(condition_method=None, data_path=Path('/Users/tommarianer/LOSC Data/gravityspy/trainingsets'), filename=None):
	"""Load labelled set for mapping.
	"""

	assert condition_method is not None or filename is not None

	if filename is None:
		filename = 'trainingset_fromraw_centered_2048_Tc_64_' + condition_method + '_split.hdf5'
	with h5py.File(join(data_path, filename), 'r') as f:
		x = np.asarray(f['x_train'])
		x = np.append(x, np.asarray(f['x_test']), axis=0)
		x = np.append(x, np.asarray(f['x_val']), axis=0)
		y = [item.decode('ascii') for item in np.asarray(f['y_train'])]
		y = np.append(y, [item.decode('ascii') for item in np.asarray(f['y_test'])], axis=0)
		y = np.append(y, [item.decode('ascii') for item in np.asarray(f['y_val'])], axis=0)
		times = np.asarray(f['times_train'])
		times = np.append(times, np.asarray(f['times_test']), axis=0)
		times = np.append(times, np.asarray(f['times_val']), axis=0)

	return x, y, times

def load_labelled_features(condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load laebelled features for mapping.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model)
	features_file = 'fromraw_' + condition_method + '_' + '_'.join(model.split('/')) + '.hdf5'
	with h5py.File(join(data_path, features_file), 'r') as f:
		features = np.asarray(f['features_train'])
		features = np.append(features, np.asarray(f['features_test']), axis=0)
		features = np.append(features, np.asarray(f['features_val']), axis=0)
		y_hat = np.asarray(f['y_train_hat'])
		y_hat = np.append(y_hat, np.asarray(f['y_test_hat']), axis=0)
		y_hat = np.append(y_hat, np.asarray(f['y_val_hat']), axis=0)
	return features, y_hat

def load_train_examples(condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load examples of training set for plotting maps.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model)
	with h5py.File(join(data_path, 'training_examples.hdf5'), 'r') as f:
		x_examples = np.asarray(f['x_examples'])
		features_examples = np.asarray(f['features_examples'])
		y_examples = [item.decode('ascii') for item in np.asarray(f['y_examples'])]
		times_examples = np.asarray(f['times_examples'])
	return x_examples, features_examples, y_examples, times_examples

def load_conditioned_images(detector, segment_file, condition_method, conditioned_path=Path('/Users/tommarianer/LOSC Data/conditioned_data/16KHZ')):
	"""Load conditioned images of specified detector and segment.
	"""

	data_path = join(conditioned_path, detector + '1/' + condition_method)
	with h5py.File(join(data_path, segment_file), 'r') as f:
		x = np.asarray(f['x'])
		times = np.asarray(f['times'])
	return x, times

def load_conditioned_features(detector, segment_file, condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load features of specified detector and segment.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model + '/' + detector + '1')
	# print(join(data_path, segment_file))
	with h5py.File(join(data_path, segment_file), 'r') as f:
		features = np.asarray(f['features'])
		y_hat = np.asarray(f['y_hat'])
		times = np.asarray(f['times'])
	return features, y_hat, times

def load_conditioned_umap(detector, segment_file, condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load features of specified detector and segment.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model + '/' + detector + '1')
	# print(join(data_path, segment_file))
	with h5py.File(join(data_path, segment_file), 'r') as f:
		features = np.asarray(f['features'])
		y_hat = np.asarray(f['y_hat'])
		times = np.asarray(f['times'])
		umap = np.asarray(f['umap'])
	return features, y_hat, times, umap

def create_map_dict(segment_file, condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Create dictionary used to plot maps.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model)
	mapper = joblib.load(join(data_path, 'mapper.sav'))
	# print(join(data_path, 'mapper.sav'))
	x_examples, features_examples, y_examples, times_examples = load_train_examples(condition_method, model, features_path)
	umap_examples = mapper.transform(features_examples)
	tomap = {'ex': {'x': x_examples, 'y': y_examples, 'umap': umap_examples, 'times': times_examples}}

	for detector in ['H', 'L']:
		x, times = load_conditioned_images(detector, segment_file, condition_method)
		y = len(times) * ['Unlabeled']
		# features, y_hat, _ = load_conditioned_features(detector, segment_file, condition_method, model, features_path)
		# umap = mapper.transform(features)
		features, y_hat, _, umap = load_conditioned_umap(detector, segment_file, condition_method, model, features_path)
		tomap[detector] = {'x': x, 'y': y, 'umap': umap, 'times': times, 'features': features}
	return tomap

def load_all_avail_features(detector, condition_method, model, features_path=Path('/Users/tommarianer/LOSC Data/gravityspy/features')):
	"""Load features of all available segments.
	"""

	data_path = join(features_path, 'fromraw_' + condition_method + '/' + model + '/' + detector + '1')
	mapper = joblib.load(join(data_path, '../mapper.sav'))
	# print(join(data_path, '../mapper.sav'))
	files = [f for f in sorted(listdir(data_path)) if isfile(join(data_path, f)) and f.split('.')[-1] == 'hdf5']
	first = True
	for file in files:
		# print(file)
		features_temp, y_hat_temp, times_temp = load_conditioned_features(detector, file, condition_method, model, features_path)
		if first:
			features = features_temp
			y_hat = y_hat_temp
			times = times_temp
			umap = mapper.transform(features)
			first = False
		else:
			features = np.append(features, features_temp, axis=0)
			y_hat = np.append(y_hat, y_hat_temp, axis=0)
			times = np.append(times, times_temp, axis=0)
			umap = np.append(umap, mapper.transform(features), axis=0)
			umap = mapper.transform(features)
	return features, y_hat, times, umap

def create_maps(data_path, mapper, replace=False):
	files = [f for f in sorted(listdir(data_path)) if isfile(join(data_path, f)) and f.split('.')[-1] == 'hdf5']
	print(len(files))
	if replace == False:
		for file in files:
			with h5py.File(join(data_path, file), 'r') as f:
				if 'umap' in f.keys():
					continue
				features = np.asarray(f['features'])
				umap = mapper.transform(features)
			with h5py.File(join(data_path, file), 'a') as f:
				f.create_dataset('umap', data=umap)
	else:
		for file in files:
			with h5py.File(join(data_path, file), 'r+') as f:
				features = np.asarray(f['features'])
				umap = mapper.transform(features)
				if 'umap' in f.keys():
					# print(np.allclose(f['umap'][()], umap))
					data = f['umap']
					data[...] = umap
				else:
					f.create_dataset('umap', data=umap)
			# with h5py.File(join(data_path, file), 'r') as f:
				# print(np.allclose(f['umap'][()], umap))
	return

def find_closest_index(t, times):
	"""Find index of image containing the time t
	"""
	
	tdiff = times - t
	idx = np.argmax(tdiff[tdiff <= 0])
	return idx

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


# the following are not used, kept them for possible future use
def compute_gda_params(features, y):
	"""Compute GDA parameters - class-conditional means and tied covariance matrix inverse (precision).
	"""
	labels = np.unique(y)
	features_mean = []
	for i, label in enumerate(labels):
		idx = [j for j, temp in enumerate(y) if temp == label]
		label_mean = np.mean(features[idx], axis=0)
		features_mean.append(label_mean)
		if i == 0:
			X = features[idx] - label_mean
		else:
			X = np.append(X, features[idx] - label_mean, axis=0)

	features_mean = np.asarray(features_mean)
	cov = EmpiricalCovariance(assume_centered=False).fit(X)
	precision = cov.precision_

	return features_mean, precision

# def compute_mahalanobis_alt(features, features_mean, precision):
# 	"""Compute minimal Mahalanobis distance between test features and each class distribution.
# 	"""
# 	M_dists = np.inf * np.ones(features.shape[0])
# 	for i in range(features_mean.shape[0]):
# 		M_c = np.diag(np.dot(features - features_mean[i], 
# 					  np.dot(precision, np.transpose(features - features_mean[i]))))
# 		M_dists = np.minimum(M_dists, M_c)
# 	return M_dists

def compute_mahalanobis(features, features_mean, precision):
	"""Compute minimal Mahalanobis distance between test features and each class distribution.
	"""
	M_dists = np.inf * np.ones(features.shape[0])
	for j, sample in enumerate(features):
		for i in range(features_mean.shape[0]):
			M_c = np.dot(sample - features_mean[i], 
						 np.dot(precision, np.transpose(sample - features_mean[i])))
			M_dists[j] = np.minimum(M_dists[j], M_c)
	return M_dists

def p_gram(features, p):
	"""Compute p-order Gram matrix of features of a single layer.
	"""
	if features.ndim == 1:
		temp = np.empty((1, features.shape[0]))
		temp[0,:] = features
	else:
		temp = features
	features = temp ** p
	G = np.dot(features, np.transpose(features)) ** (1/p)
	G_bar = G[np.triu_indices(G.shape[0])]
	return G_bar

def mins_maxs(features, y, p_list):
	labels = np.unique(y)
	if features[0].ndim == 1:
		n_l = 1
	else:
		n_l = features.shape[1]
	mins = np.inf * np.ones((len(labels), len(p_list), int(0.5 * n_l * (n_l + 1))))
	maxs = -np.inf * np.ones((len(labels), len(p_list), int(0.5 * n_l * (n_l + 1))))
	for c, label in enumerate(labels):
		idx = [j for j, temp in enumerate(y) if temp == label]
		for sample in features[idx]:
			for j, p in enumerate(p_list):
				G_bar = p_gram(sample, p)
				mins[c,j,:] = np.minimum(mins[c,j,:], G_bar)
				maxs[c,j,:] = np.maximum(maxs[c,j,:], G_bar)
	return mins, maxs

def deviations(features, y, labels, p_list, mins, maxs):
	delta = np.zeros((features.shape[0], len(p_list), mins.shape[-1]))
	for i in range(features.shape[0]):
		c = np.where(labels == y[i])[0][0]
		for j, p in enumerate(p_list):
			G_bar = p_gram(features[i], p)
			for k, val in enumerate(G_bar):
				if val < mins[c,j,k]:
					delta[i,j,k] = (mins[c,j,k] - val) / np.abs(mins[c,j,k])
				elif val > maxs[c,j,k]:
					delta[i,j,k] = (val - maxs[c,j,k]) / np.abs(maxs[c,j,k])
	delta = np.sum(np.sum(delta, axis=-1), axis=-1)
	return delta