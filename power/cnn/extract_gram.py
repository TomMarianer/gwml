#!/usr/bin/python
"""
script for extracting latent space features for the training set spectrograms
it also extracts predictions and gram matrix method values - mins and maxs for each class, 
and deviations for the test and validation sets (including mean validation deviation)
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

import sys
# sys.path.append('/dovilabfs/work/tommaria/gw/tools')
sys.path.append(git_path + '/power/tools')
from gstools import *
from gsparams import *
from keras import backend as K

# Load models

start_time = time.time()

model_path = Path('/dovilabfs/work/tommaria/gw/gravityspy/gpu/' + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name)
model_file = join(model_path, model_name + '.h5')
weights_file = join(model_path, model_name + '.weights.best.hdf5')

model = load_model(model_file)
model.load_weights(weights_file)

inp = model.input                                          							   # input placeholder
outputs = [layer.output for layer in model.layers if layer_conditions(layer)]          # all layer outputs
outputs.append(model.layers[-2].output)
outputs.append(model.layers[-1].output)
functor = K.function([inp, K.learning_phase()], outputs)   						   # evaluation function

# Load datasets

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')

with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_split_wrap_no_ty.hdf5'), 'r') as f:
	x_train = np.asarray(f['x_train'])
	x_test = np.asarray(f['x_test'])
	x_val = np.asarray(f['x_val'])
	y_train = np.asarray(f['y_train'])
	y_test = np.asarray(f['y_test'])
	y_val = np.asarray(f['y_val'])
	times_train = np.asarray(f['times_train'])
	times_test = np.asarray(f['times_test'])
	times_val = np.asarray(f['times_val'])

if x_train.shape[1:3] != input_size:
	x_train = np.asarray([resize(img, input_size, mode='reflect') for img in x_train])
	x_test = np.asarray([resize(img, input_size, mode='reflect') for img in x_test])
	x_val = np.asarray([resize(img, input_size, mode='reflect') for img in x_val])

p_list = np.arange(1,2)
print('p list: ' + str(p_list))
mins, maxs, features_train, y_train_hat = mins_maxs_new(x_train, y_train, p_list, functor)

print("--- min-max time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
start_time = time.time()

labels = np.unique(y_train)
delta_test, features_test, y_test_hat = deviations_features_predictions_new(x_test, labels, p_list, mins, maxs, functor)
delta_val, features_val, y_val_hat = deviations_features_predictions_new(x_val, labels, p_list, mins, maxs, functor)

delta_mean_test_val = np.mean(np.append(delta_test, delta_val, axis=0), axis=0)

print(features_train.shape)
print(y_train_hat.shape)
print(delta_test.shape)
print(delta_val.shape)
print(delta_mean_test_val.shape)

print("--- test-val mean dev time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))

start_time = time.time()

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/features/' + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name + '/gram')
if not exists(data_path):
	makedirs(data_path)

with h5py.File(join(data_path, train_set + '_new_' + extract_model + '_' + opt_method + '_' + model_name + '.hdf5'), 'w') as f:
	f.create_dataset('features_train', data=features_train)
	f.create_dataset('features_test', data=features_test)
	f.create_dataset('features_val', data=features_val)
	f.create_dataset('y_train_hat', data=y_train_hat)
	f.create_dataset('y_test_hat', data=y_test_hat)
	f.create_dataset('y_val_hat', data=y_val_hat)
	f.create_dataset('times_train', data=times_train)
	f.create_dataset('times_test', data=times_test)
	f.create_dataset('times_val', data=times_val)
	f.create_dataset('delta_test', data=delta_test)
	f.create_dataset('delta_val', data=delta_val)
	f.create_dataset('delta_mean', data=delta_mean_test_val)
	grp = f.create_group('mins')
	for c, c_vals in enumerate(mins):
		for l, l_vals in enumerate(c_vals):
			grp.create_dataset(str(c) + '_' + str(l), data=l_vals)
	
	grp = f.create_group('maxs')
	for c, c_vals in enumerate(maxs):
		for l, l_vals in enumerate(c_vals):
			grp.create_dataset(str(c) + '_' + str(l), data=l_vals)

print("--- save time is %.7s seconds ---\n" % (time.time() - start_time))
