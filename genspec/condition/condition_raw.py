#!/usr/bin/env python3
# script for conditioning raw strain data and generating spectrograms
# non parallel version

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append('../tools')
# from tools import *
# from tools_alternate import *
from tools_gs import *
# from tools_gs_par import *

segment_list = get_segment_list('BOTH')
detector = 'L'

files = get_files(detector)

# Old process:
# segment = segment_list[4]
# # H1 segments - 0,1,2,3,4
# # L1 segments - 0,1,2,3,4

# # New process (alt):
# segment = segment_list[3]
# # H1 segments - 0,1,2,3,4
# # L1 segments - 0,1,2,3,4

# New process (gs_based):
# seg_num = 1659
# Tc=16:
# H1 segments - 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21-29,30-39,40-49,50-59,60-69,70-79,325-326,586-588,660-669,690-699,700-709,710-719,720-729,811,1587,1633,1650,1659,1669,1687-part
# L1 segments - 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21-29,30-39,40-49,50-59,60-69,70-79,325-326,586-588,660-669,690-699,700-709,710-719,720-729,811,1587,1633,1650,1659,1669,1687-part
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

	# start_time = time.time()

	# condition_chunks(t_i, t_f, Tc=16, To=2, local=True, fw=4*4096, qtrans=True, qsplit=True, dT=2, detector=detector, save=True, data_path=data_path)
	condition_chunks(t_i, t_f, Tc=64, To=2, local=True, fw=4*4096, qtrans=True, qsplit=True, dT=2, detector=detector, save=True, data_path=data_path)

print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
