#!/usr/bin/python
"""
script for extracting latent space features for unlabeled spectrograms
including gram matrix method deviations and total deviation normalized by the mean validation devation
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

# Load mins, maxs

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')
with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_split_wrap_no_ty.hdf5'), 'r') as f:
	y_train = np.asarray(f['y_train'])

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/features/' + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name + '/gram')
with h5py.File(join(data_path, train_set + '_new_' + extract_model + '_' + opt_method + '_' + model_name + '.hdf5'), 'r') as f:
	mins = load_group(f['mins'])
	maxs = load_group(f['maxs'])
	delta_mean = np.asarray(f['delta_mean'])

inp = model.input                                          							   # input placeholder
outputs = [layer.output for layer in model.layers if layer_conditions(layer)]          # all layer outputs
outputs.append(model.layers[-2].output)
outputs.append(model.layers[-1].output)
functor = K.function([inp, K.learning_phase()], outputs)   						   # evaluation function
labels = np.unique(y_train)
p_list = np.arange(1,2)

detector = 'H'

cond_data_path = Path('/scratch300/tommaria/data/conditioned_data/16KHZ/' + detector + '1/')
files = [f for f in sorted(listdir(cond_data_path)) if isfile(join(cond_data_path, f))]
	
for file in files:
	with h5py.File(join(cond_data_path, file), 'r') as f:
		x = np.asarray(f['x'])
		times = np.asarray(f['times'])

	print(file)

	if x.shape[1:3] != input_size:
		x = np.asarray([resize(img, input_size, mode='reflect') for img in x])

	delta, features, y_hat = deviations_features_predictions_new(x, labels, p_list, mins, maxs, functor)
	total_dev = np.dot(delta, 1 / delta_mean)

	print(delta.shape)
	print(features.shape)
	print(y_hat.shape)
	print(total_dev.shape)

	data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/features/' 
					 + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name + '/gram/' + detector + '1/injected')
	
	if not exists(data_path):
		makedirs(data_path)

	with h5py.File(join(data_path, file), 'w') as f:
		f.create_dataset('features', data=features)
		f.create_dataset('y_hat', data=y_hat)
		f.create_dataset('times', data=times)
		f.create_dataset('delta', data=delta)
		f.create_dataset('total_dev', data=total_dev)

print(detector)
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
