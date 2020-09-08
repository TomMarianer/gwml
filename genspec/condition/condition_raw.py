#!/usr/bin/env python3
# script for conditioning raw strain data and generating spectrograms
# non parallel version

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
from tools_gs import *
from params import *

segment_list = get_segment_list('BOTH')
detector = 'L'

files = get_files(detector)

start_time = time.time()
for seg_num in np.arange(42,45): #np.arange(40,50):
	segment = segment_list[seg_num]
	# Tc=64:
	# H1 segments - 0-3,4,5-8,9,10-19,20-29,326,587,811,1587,1633,1650,1659,1669,1687
	# L1 segments - 0-3,4,5-8,9,10-19,20-29,326,587,811,1587,1633,1650,1659,1669,1687


	t_i = segment[0]
	t_f = segment[1]

	data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + 
	                 '1/segment-' + str(t_i) + '-' + str(t_f))

	print(seg_num)
	print(data_path)

	condition_chunks(t_i, t_f, Tc=Tc, To=To, local=True, fw=fw, qtrans=True, qsplit=True, dT=dT, detector=detector, save=True, data_path=data_path)

print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
