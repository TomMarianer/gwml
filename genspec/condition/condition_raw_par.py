#!/usr/bin/env python3
"""
script for conditioning raw strain data and generating spectrograms
parallel version
"""

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
from tools_gs_par import *
from params import *

segment_list = get_segment_list('BOTH')
detector = 'H'

files = get_files(detector)
pool = mp.Pool(mp.cpu_count() - 1)

start_time = time.time()
for seg_num in np.arange(400,450):
	segment = segment_list[seg_num]
	# Tc=64:
	# H1 segments - 0-3,4,5-8,9,10-19,20-29,30-39,40-49,50-59,60-69,70-99,100-149,150-199,200-249,250-299,300-325,326,327-349,350-399,548,587,590-639,640-689,690-739,740-789,811,877,921,1117,1265,1369,1580,1587,1633,1650,1659,1669,1687
	# L1 segments - 0-3,4,5-8,9,10-19,20-29,30-39,40-49,50-59,60-69,70-99,100-149,150-199,200-249,250-299,300-325,326,327-349,350-399,548,587,590-639,640-689,690-739,740-789,811,877,921,1117,1265,1369,1580,1587,1633,1650,1659,1669,1687


	t_i = segment[0]
	t_f = segment[1]

	data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + 
	                 '1/segment-' + str(t_i) + '-' + str(t_f))

	print(seg_num)
	print(data_path)

	local = True
	qtrans = True
	qsplit = True
	save = True

	chunks = get_chunks(t_i, t_f, Tc, To)
	results = pool.starmap(load_condition_save, [(chunk[0], chunk[1], local, Tc, To, fw, window, detector, qtrans, qsplit, dT, save, data_path) for chunk in chunks])
	
pool.close()
pool.join()


print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
