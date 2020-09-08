#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')

import sys
sys.path.append('/storage/home/tommaria/thesis/tools')
# from tools import *
# from tools_alternate import *
from tools_gs import *

segment_list = get_segment_list('BOTH')
detector = 'L'
print(detector)

# Old process:
# H1 segments - 0,1
# L1 segments - 0,1,2,3,4

# New process (alt):
# H1 segments - 0-4
# L1 segments - 0-4

# GS based process:
# Tc=16
# H1 segments - 0-4,5-18,19-29,30-39,325-326,586-588,660-669,690-709,710-729,811,1587,1633,1650,1659,1669,1687-part
# L1 segments - 0-4,5-18,19-29,30-39,325-326,586-588,660-669,690-709,710-729,811,1587,1633,1650,1659,1669,1687-part

# Tc=64
# H1 segments - 0-4,5-8,9-49,50-99,100-149,150-199,200-249,250-299,300-325,326,327-349,350-399,548,587,590-639,640-689,690-739,740-789,811,877,921,1117,1265,1369,1580,1587,1633,1650,1659,1669,1687
# L1 segments - 0-4,5-8,9-49,50-99,100-149,150-199,200-249,250-299,300-325,326,327-349,548,587,590-639,640-689,690-739,740-789,811,877,921,1117,1265,1369,1580,1587,1633,1650,1659,1669,1687

start_time = time.time()

# for segment in segment_list[40:80]:
for seg_num in np.arange(362,400):
	segment = segment_list[seg_num]

	print(seg_num, segment)

	t_i = segment[0]
	t_f = segment[1]

	data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + 
	                 '1/segment-' + str(t_i) + '-' + str(t_f))

	files = [join(data_path, f) for f in sorted(listdir(data_path)) if isfile(join(data_path, f))]

	x = []
	times = []

	for file in files:
		with h5py.File(join(data_path,file), 'r') as f:
			x.extend(np.asarray(f['values']))
			times.extend(np.asarray(f['t0']))

	x = np.asarray(x)
	times = np.asarray(times)

	data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/combined')
	if not exists(data_path):
		makedirs(data_path)

	with h5py.File(join(data_path, 'segment-' + str(t_i) + '-' + str(t_f) + '.hdf5'), 'w') as f:
		f.create_dataset('x', data=x)
		f.create_dataset('times', data=times)

	print(x.shape)
	print(times.shape)

print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
