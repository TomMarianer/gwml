#!/usr/bin/python
"""
script for evaluating the trained network
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

# Load dataset

data_path = Path('/dovilabfs/work/tommaria/gw/data/gravityspy/fromraw')
with h5py.File(join(data_path, 'trainingset_fromraw_centered_2048_Tc_64_gs_split_wrap_no_ty.hdf5'), 'r') as f:
	x_train = np.asarray(f['x_train'])
	x_test = np.asarray(f['x_test'])
	x_val = np.asarray(f['x_val'])
	y_train = [item.decode('ascii') for item in list(f['y_train'])]
	y_test = [item.decode('ascii') for item in list(f['y_test'])]
	y_val = [item.decode('ascii') for item in list(f['y_val'])]

# Encode the labels

labels = np.unique(y_train)

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

# Load model

start_time = time.time()

model_path = Path('/dovilabfs/work/tommaria/gw/gravityspy/gpu/' + train_set + '/new/' + extract_model + '/' + opt_method + '/' + model_name)
weights_file = join(model_path, model_name + '.weights.best.hdf5')
model = load_model(join(model_path, model_name + '.h5'))
model.load_weights(weights_file)
model.summary()

# Evaluate accuracy

batch_size = 32

print('Model: ' + opt_method + '/' + extract_model + '/' + model_name + '\n')

# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print('Test set - loss: %f, accuracy: %f\n' %(score[0], score[1]))

score = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
print('Val set - loss: %f, accuracy: %f\n' %(score[0], score[1]))

score = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
print('Train set - loss: %f, accuracy: %f\n' %(score[0], score[1]))

print("--- Execution time is %.7s seconds ---\n" % (time.time() - start_time))
