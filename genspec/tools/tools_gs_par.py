#!/usr/bin/python
"""
tools for loading and conditioning strain data, and generating spectrograms
parallel version
"""

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.signal import filter_design
from gwpy.segments import Segment
import scipy.signal as scisig
import numpy as np
import time
import astropy.units as units
from os import listdir, makedirs
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from skimage.transform import resize
from pathlib import Path
import h5py
import multiprocessing as mp

def get_segment_list(detector='H'):
	"""
	this function reads the segment list of a given detector
	input: detector - string with detector name, can be 'H', 'L' or 'BOTH'
	output: segment_list - list of segments of times when the given detector was active
	"""

	data_path = Path('../segment_files')
	if detector == 'BOTH':
		# filename = join(data_path, 'O1_' + detector + '_DATA')
		filename = join(data_path, 'O1_O2_' + detector + '_DATA')
	else:
		filename = join(data_path, 'O1_' + detector + '1_DATA')
	f = open(filename,'r')
	segment_list = []
	for line in f:
		line_str = line.split(' ')
		segment = (int(line_str[0]),int(line_str[1]))
		segment_list.append(segment)
	f.close()
	return segment_list

def get_files(detector='H'):
	"""
	this function returns a list of local data files for given detector
	input: detector - string with detector name, can be 'H' or 'L'
	output: files - list of file paths with local strain data
	"""

	# data_path = Path('/storage/fast/users/tommaria/data/bulk_data/16KHZ/' + detector + '1')
	data_path = Path('/arch/tommaria/data/bulk_data/16KHZ/' + detector + '1')
	files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
	return files

def get_chunks(t_i, t_f, Tc=16, To=2):
	"""
	this function receives initial and final times and divides it into chunks of given length and overlap
	inputs:
		t_i - int - initial gps time in seconds
		t_f - int - final gps time in seconds
		Tc - int - chunk length in seconds, default=8
		To - float - overlap time between chunks, default=1
	output: chunks - list of chunk initial and final times
	"""

	if t_i + Tc > t_f:
		return [(t_i, t_f)]

	chunks = []
	c_i = t_i
	while (c_i + Tc) <= t_f:
		chunks.append((c_i, c_i + Tc))
		c_i += (Tc - To)

	# add the final modified chunk to adjust the segment size
	# (if the segment didn't divide to an integer number of chunks)
	if chunks[-1][1] < t_f:
		chunks.append((t_f - Tc, t_f))
	return chunks

class QTImage:
	"""
	Q-Transform image object
	"""
	def __init__(self, values, t0, times, f0, frequencies):
		self.values = values
		self.t0 = t0
		self.times = times
		self.f0 = f0
		self.frequencies = frequencies

def get_log_freqs(f_i, f_f, length):
	"""
	this function gets the log frequencies TODO: ADD DOC
	"""
	frequencies = np.logspace(np.log10(f_i), np.log10(f_f), length)
	return frequencies
        
def qimg_draw(qt, vmin=0, vmax=None, input_size=(299, 299)):
	"""
	this function receives a q-transform and returns the rgb image made by plotting it
	inputs:
		qt - Spectrogram - 2darray with q-transform values
		vmin, vmax - scalar - the value range for the colormap to cover. if None the full range will be covered.
			if vmax='auto' the range will be set automatically according to the q-transform values
		input_size - tuple - required size of qimg values, default=299x299
	output:
		qimg - QTImage - QTImage object with the q-transform image
	"""

	# plot = plt.figure(figsize=(10, 10))

	# if vmax == 'auto':
	# 	vmax = qt.max()
	# if vmax > 500:
	# 	vmax = vmax / 10

	plot = qt.plot(figsize=(10, 10), vmin=vmin, vmax=vmax)
	ax = plot.gca()
	# ax = plot.add_subplot(1, 1, 1)
	# ax.plot(qt, vmin=vmin, vmax=vmax)
	ax.grid(False)
	ax.set_yscale('log')

	plot.subplots_adjust(bottom = 0)
	plot.subplots_adjust(top = 1)
	plot.subplots_adjust(right = 1)
	plot.subplots_adjust(left = 0)

	ax.set_axis_off()
	extent = ax.get_window_extent().transformed(plot.dpi_scale_trans.inverted())

	plot.canvas.draw()
	values = np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8)
	values = values.reshape(plot.canvas.get_width_height()[::-1] + (3,))
	values = np.flipud(values)
	plot.close()
	values = resize(values, input_size, mode='reflect')
	values = values[::-1,:,:]

	frequencies = get_log_freqs(qt.f0.value, qt.frequencies[-1].value, values.shape[0])
	times = np.linspace(qt.t0.value, qt.times[-1].value, values.shape[1])
	qimg = QTImage(values, qt.t0.value, times, qt.f0.value, frequencies)
	return qimg

def qimg_split(qt, dT=2.0, vmin=0, vmax=None):
	"""
	this function receives a q-transform, splits it to smaller time segments
	and returns a list with q-transform images for each smaller segment
	inputs:
		qt - Spectrogram - 2darray with q-transform values
		dT - float - length in time for each q-transform image
		vmin, vmax - scalar - the value range for the colormap to cover. if None the full range will be covered.
			if vmax='auto' the range will be set automatically according to the q-transform values
	output:
		qimg - list with q-transform images
	"""

	t_i = qt.t0.value
	t_f = qt.times[-1].value
	T = t_i
	qimg = []
	while T + dT <= t_f + np.max(np.diff(qt.times.value)):
		qcrop = qt.crop(T, T + dT)
		qimg.append(qimg_draw(qcrop, vmin=vmin, vmax=vmax))
		T += dT
	return qimg

def img_qtransform(data, To=2, frange=(10, 2048), qrange=(4, 64), vmin=0, qsplit=False, dT=2.0):
	"""
	this funciton performs the q-transform on the strain data and returns it as an rgb image
	inputs:
		data - TimeSeries - the strain data to be transformed
		frange - tuple - frequency range of the q-transform
		qrange - tuple - q factor range of the q-transform
		vmin - scalar - the value range for the colormap to cover. if None the full range will be covered.
			if vmax='auto' the range will be set automatically according to the q-transform values
		qsplit - if True split the qtransform to separate images for better resolution
	output:
		qimg - ndarray - array with the q-transform image
	"""

	# qt = data.q_transform(frange=frange, qrange=qrange, whiten=True, tres=0.002) # compute the q-transform with the built in function of gwpy
	# qt = qt[int(To/2.0 / qt.dt.value):-int(To/2.0 / qt.dt.value)] # crop the q-transform to remove half the overlap time at each end
	# vmax = qt.max()
	# print(vmax.value)
	vmax = 25.5 # I think this is the value used in gravityspy (if I understand correctly)
	if qsplit:
		qt = data.q_transform(frange=frange, qrange=qrange, whiten=True, tres=0.002) # compute the q-transform with the built in function of gwpy
		qt = qt[int(To/2.0 / qt.dt.value):-int(To/2.0 / qt.dt.value)] # crop the q-transform to remove half the overlap time at each end
		qimg = qimg_split(qt, dT=dT, vmin=vmin, vmax=vmax)
	else:
		t_center = data.times[int(len(data.times)/2)].value
		outseg = Segment(t_center - dT/2, t_center + dT/2)
		qt = data.q_transform(frange=frange, qrange=qrange, whiten=True, tres=0.002, gps=t_center, search=0.5, fres=0.5, outseg=outseg)
		qt = qt.crop(t_center - dT/2, t_center + dT/2)
		qimg = qimg_draw(qt, vmin, vmax)
	return qimg

def split_qtransform(data, To=2, frange=(10, 2048), qrange=(4, 100), vmin=0, qsplit=False, dT=2.0):
	"""
	"""

	qt_buffer = 1 # 1 second buffer for computing qtransform, since gwpy uses some window to compute qtransform
	# vmax = 25 # I think this is the value used in gravityspy (if I understand correctly)
	# vmax = 30 # right now this is the value that creates images as close to the gravity spy ones as I got.
	vmax = None # try with no clipping, not sure it preserves the data but I think it does, make sure (but for now can start the long process of creating data)

	t_i = data.t0.value + To/2.0
	t_f = data.times[-1].value - To/2.0
	T = t_i
	qimg = []
	while T + dT <= t_f + np.max(np.diff(data.times.value)):
		qt = data.crop(T - qt_buffer, T + dT + qt_buffer).q_transform(frange=frange, qrange=qrange, whiten=False, tres=0.002) # crop and compute qtransform
		qcrop = qt.crop(T, T + dT)
		qimg.append(qimg_draw(qcrop, vmin=vmin, vmax=vmax))
		T += dT

	return qimg	


def condition_data(data, To=2, fw=2048, window='tukey', qtrans=False, qsplit=False, dT=2.0):
	"""
	this functions conditions the data in a similar manner to what is done in th 'Omicron' algorithm
	inputs:
		data - TimeSeries - the data to be conditioned
		To - float - overlap time between chunks, default=1
		fw - int - working frequency in Hz, default=2048
		window - ndarray or string - window to use, defualt='tukey' - use tukey window
		qtrans - if True perform the qtransform and return that as the conditioned data. default=False
		qsplit - if True split the qtransform to separate images for better resolution
		dT - float - length in time for each q-transform image
	output:
		cond_data - TimeSeries or ndarray - conditioned data, either strain data TimeSeries or ndarray with qtransform image
	"""

	cond_data = data - data.mean() # remove DC component
	# cond_data = cond_data.resample(rate=fw, ftype = 'iir', n=20) # downsample to working frequency fw
	# cond_data = cond_data.resample(rate=4096, ftype = 'iir', n=20) # downsample to working frequency fw
	cond_data = cond_data.highpass(frequency=20, filtfilt=True) # filter out frequencies below 20Hz

	Nc = len(cond_data)
	Tc = Nc * cond_data.dt.value
	if window == 'tukey':
		window = scisig.tukey(M=Nc, alpha=1.0*To/Tc, sym=True)

	cond_data = cond_data * window
	# print(sum(cond_data**2 * cond_data.dt.value))
	# cond_data = cond_data.whiten(fftlength=2, overlap=1)
	# print(sum(cond_data**2 * cond_data.dt.value))

	if qtrans:
		cond_data = img_qtransform(cond_data, To, qsplit=qsplit, dT=dT) #, frange=(8, fw/2))
		# cond_data = split_qtransform(cond_data, To, qsplit=qsplit, dT=dT) #, frange=(8, fw/2))

	return cond_data

def condition_chunks(t_i, t_f, local=False, Tc=16, To=2, fw=2048, window='tukey', detector='H', 
					 qtrans=False, qsplit=False, dT=2.0, save=False, data_path=None):
	"""
	this functions conditions the data between a given initial and final time by first dividing it into chunks
	and then conditioning each chunk. the initial and final times must be within the same segment
	inputs:
		local - bool - if True load strain data from local files, else load strain data directly from gwosc
			default=True
		t_i - int - initial gps time in seconds
		t_f - int - final gps time in seconds
		Tc - int - chunk length in seconds, default=8
		To - float - overlap time between chunks, default=1
		fw - int - working frequency in Hz, default=2048
		window - ndarray or string - window to use, defualt='tukey' - use tukey window
		detector - string - choose which detector to work on, can be 'H' or 'L'
		qtrans - if True perform the qtransform and return that as the conditioned data. default=False
		qsplit - if True split the qtransform to separate images for better resolution
		dT - float - length in time for each q-transform image
		save - if True save the conditioned data to hdf5 file
		data_path - path to save hdf5 files in. default=None and then path is hard coded in the function
	output:
		cond_data_list - list with TimeSeries of conditioned data or ndarrays of qimages (if qtrans=True)
	"""

	# pool = mp.Pool(mp.cpu_count())
	pool = mp.Pool(16)

	chunks = get_chunks(t_i, t_f, Tc, To)
	cond_data_list = []

	# results = [pool.apply(load_condition_save, args=(chunk[0], chunk[1], local, Tc, To, fw, window, detector, qtrans, qsplit, dT, save, data_path)) for chunk in chunks]
	
	results = []
	for chunk in chunks:
		pool.apply_async(wrapped, args=(chunk[0], chunk[1], local, Tc, To, fw, window, detector, qtrans, qsplit, dT, save, data_path), callback=collect_results)

	pool.close()
	pool.join()

	if save:
		return
	elif qsplit and qtrans:
		for res in results:
			cond_data_list.extend(res)
	else:
		cond_data_list = results
	
	return cond_data_list

def collect_results(result):
	global results
	results.append(result)

def wrapped(t_i, t_f, local=False, Tc=16, To=2, fw=2048, window='tukey', detector='H', 
						qtrans=False, qsplit=False, dT=2.0, save=False, data_path=None):
	try:
		load_condition_save(t_i, t_f, local, Tc, To, fw, window, detector, qtrans, qsplit, dT, save, data_path)
	except:
		print('Excetption: (%d, %d)' % (t_i, t_f))

def load_condition_save(t_i, t_f, local=False, Tc=16, To=2, fw=2048, window='tukey', detector='H', 
						qtrans=False, qsplit=False, dT=2.0, save=False, data_path=None):
	"""Fucntion to load condition and save chunk, created to enable parallelizing.
	"""

	conditioned_files = []
	if exists(data_path):
		conditioned_files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
		# print(len(conditioned_files))
	fname = 'conditioned-chunk-' + str(t_i) + '-' + str(t_f) + '.hdf5'
	if join(data_path, fname) in conditioned_files:
		# print(fname)
		return

	if local:
		files = get_files(detector)
		try:
			data = TimeSeries.read(files, start=t_i, end=t_f, format='hdf5.losc') # load data locally
		except:
			return

	else:
		# load data from losc
		try:
			data = TimeSeries.fetch_open_data(detector + '1', *(t_i, t_f), sample_rate=fw, verbose=False, cache=True)
		except:
			return

	if np.isnan(data.value).any():
		return

	cond_data = condition_data(data, To, fw, window, qtrans, qsplit, dT)
	if save:
		values = []
		t0 = []
		times = []
		f0 = []
		frequencies = []
		for dat in cond_data:
			values.append(dat.values)
			t0.append(dat.t0)
			times.append(dat.times)
			f0.append(dat.f0)
			frequencies.append(dat.frequencies)

		values = np.asarray(values)
		t0 = np.asarray(t0)
		times = np.asarray(times)
		f0 = np.asarray(f0)
		frequencies = np.asarray(frequencies)

		if data_path == None:
			data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1')

		if not exists(data_path):
			makedirs(data_path)

		fname = 'conditioned-chunk-' + str(t_i) + '-' + str(t_f) + '.hdf5'
		with h5py.File(join(data_path,fname), 'w') as f:
			f.create_dataset('values', data=values)
			f.create_dataset('t0', data=t0)
			f.create_dataset('times', data=times)
			f.create_dataset('f0', data=f0)
			f.create_dataset('frequencies', data=frequencies)

		return

	else:
		return cond_data

def condition_segments(segment_list, local=False, Tc=16, To=2, fw=2048, window='tukey', detector='H', 
					   qtrans=False, qsplit=False, dT=2.0, save=False):
	"""
	this function performs the data conditoining on several segments
	inputs:
		segment_list - list of segments to perform the conditioning on
		local - bool - if True load strain data from local files, else load strain data directly from gwosc
			default=True
		Tc - int - chunk length in seconds, default=8
		To - float - overlap time between chunks, default=Tc/8
		fw - int - working frequency in Hz, default=2048
		window - ndarray or string - window to use, defualt='tukey' - use tukey window
		detector - string - choose which detector to work on, can be 'H' or 'L'
		qtrans - if True perform the qtransform and return that as the conditioned data. default=False
		qsplit - if True split the qtransform to separate images for better resolution
		dT - float - length in time for each q-transform image
		save - if True save the conditioned data to hdf5 file
	output:
		cond_data_list - list with TimeSeries of conditioned data or ndarrays of qimages (if qtrans=True)
	"""

	cond_data_list = []
	for segment in segment_list:
		cond_data_list.extend(condition_chunks(segment[0], segment[1], local, Tc, To, fw, window, detector, qtrans, qsplit, dT))
	return cond_data_list

def find_closest_index(t, times):
	"""Find index of image containing the time t
	"""
	
	tdiff = times - t
	idx = np.argmax(tdiff[tdiff < 0])
	return idx
