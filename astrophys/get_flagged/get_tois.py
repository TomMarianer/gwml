#!/usr/bin/python
"""
loading the time-stamps (times of interest - tois) flagged as outliers by each method
it is actually quite reduntant to keep this in a separate file now, but previously it was warranted
so in order to not make changes to get_pois_script.py it is still in a separate file
"""

import h5py
import numpy as np
from os.path import join
import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
from tools_gs_par import *

def get_tois(method):
	with h5py.File('../../shared/tois.hdf5', 'r') as f:
		tois = np.asarray(f[method])

	return tois