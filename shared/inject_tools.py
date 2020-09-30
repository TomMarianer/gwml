#!/usr/bin/python

import git
import numpy as np
import pandas as pd
import psutil
import gc
import sys
from os.path import isfile, join, exists, dirname, realpath
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal import chirp, iirdesign, filtfilt
from numpy.random import randn
from pycbc.detector import Detector

def get_git_root(path):
	"""Get git root path
	"""
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root

file_path = dirname(realpath(__file__))
git_path = get_git_root(file_path)

import sys
sys.path.append(git_path + '/astrophys/tools')
from tools_gs_par import *

def gaussian_env(times, t_inj, tau):
	"""Generate gaussian envelope.
	"""
	env = np.exp(-(times - t_inj) ** 2 / tau ** 2)
	return env

def sine_gaussian(times, t_inj, f0, Q):
	"""Generate sine-gaussian waveform.
	"""
	tau = Q / (np.sqrt(2) * np.pi * f0)
	env = gaussian_env(times, t_inj, tau)
	Hp = env * np.sin(2 * np.pi * f0 * (times - t_inj))
	Hc = env * np.cos(2 * np.pi * f0 * (times - t_inj))
	return Hp, Hc

def gaussian(times, t_inj, tau):
	"""Generate gaussian waveform.
	"""
	Hp = gaussian_env(times, t_inj, tau)
	Hc = np.zeros(Hp.shape)
	return Hp, Hc

def ringdown(times, t_inj, f0, tau):
	"""Generate ringdown waveform.
	"""
	env = np.exp(-(times - t_inj) / tau) * np.heaviside(times - t_inj, 0)
	env[np.isnan(env)] = 0
	Hp = env * np.sin(2 * np.pi * f0 * (times - t_inj))
	Hc = env * np.cos(2 * np.pi * f0 * (times - t_inj))
	return Hp, Hc

def chirp_gaussian(times, t_inj, f0, t1, f1, Q, method='linear'):
	"""Generate chirp-gaussian waveform.
	"""
	tau = Q / (np.sqrt(2) * np.pi * f0)
	env = gaussian_env(times, t_inj, tau)
	Hp = env * chirp(times - t_inj, f0, t1, f1, method=method)
	Hc = env * chirp(times - t_inj, f0, t1, f1, method=method, phi=90)
	return Hp, Hc

def chirp_gaussian_inc(times, t_inj, f0, t1, f1, Q, method='linear'):
	"""Generate chirp-gaussian waveform.
	"""
	tau = Q / (np.sqrt(2) * np.pi * f0)
	env = gaussian_env(times, t_inj, tau)
	Hp = env * chirp(times - t_inj + np.sqrt(2) * tau, f0, t1, f1, method=method)
	Hc = env * chirp(times - t_inj + np.sqrt(2) * tau, f0, t1, f1, method=method, phi=90)
	return Hp, Hc

def double_chirp_gaussian(times, t_inj, f0, t1, f1, a, f2, Q, method='linear'):
	"""Generate sine-gaussian-chirp waveform.
	"""
	tau = Q / (np.sqrt(2) * np.pi * f0)
	env = gaussian_env(times, t_inj, tau)
	Hp = env * (chirp(times - t_inj + tau, f0, t1, f1, method=method) + 
				a * chirp(times - t_inj + tau, f0, t1, f2, method=method))
	Hc = env * (chirp(times - t_inj + tau, f0, t1, f1, method=method, phi=90) + 
				a * chirp(times - t_inj + tau, f0, t1, f2, method=method, phi=90))
	return Hp, Hc

def white_noise(times, t_inj, f_low, f_high, tau, fs=4096*4):
	"""Generate white noise waveform
	"""
	env = gaussian_env(times, t_inj, tau)
	sig_p = randn(len(times))
	sig_c = randn(len(times))
	b, a = iirdesign(wp=(f_low, f_high), ws=(f_low * 2/3., min(f_high * 1.5, fs/2.)), gpass=2, gstop=30, fs=fs)
	Hp = env * filtfilt(b, a, sig_p)
	Hc = env * filtfilt(b, a, sig_c)
	return Hp, Hc

def gen_waveform(A, alpha, Hp, Hc):
	hp = A * (1 + alpha ** 2) / 2 * Hp
	hc = A * alpha * Hc
	return hp, hc

def find_segment(t, segment_list):
	"""Find segment time t is in
	"""
	for segment in segment_list:
		if segment[0] <= t and t <= segment[1]:
			return segment
	return None

def gen_inject(wf_times, dt, t_inj, alpha, inj_type, inj_params, Tc, fw):
	"""Inject waveform to data
	"""

	if inj_type == 'sg':
		Hp, Hc = sine_gaussian(wf_times, t_inj, inj_params['f0'], inj_params['Q'])

	elif inj_type == 'ga':
		Hp, Hc = gaussian(wf_times, t_inj, inj_params['tau'])

	elif inj_type == 'rd':
		Hp, Hc = ringdown(wf_times, t_inj, inj_params['f0'], inj_params['tau'])

	elif inj_type == 'cg':
		Hp, Hc = chirp_gaussian(wf_times, t_inj, inj_params['f0'], inj_params['Q'] / (np.sqrt(2) * np.pi * inj_params['f0']), 
								inj_params['f0'] / 2, inj_params['Q'], method='linear')

	elif inj_type == 'cg_inc':
		Hp, Hc = chirp_gaussian_inc(wf_times, t_inj, inj_params['f0'], inj_params['Q'] / (np.sqrt(2) * np.pi * inj_params['f0']), 
									inj_params['f0'] * 20, inj_params['Q'], method='linear')

	elif inj_type == 'cg_double':
		if inj_params['f0'] == 30:
			a = 0.6
			f2 = inj_params['f0'] * 2
		elif inj_params['f0'] == 70:
			a = 0.2
			f2 = inj_params['f0'] * 1.2

		Hp, Hc = double_chirp_gaussian(wf_times, t_inj, inj_params['f0'], inj_params['Q'] / (np.sqrt(2) * np.pi * inj_params['f0']), 
									inj_params['f0'] * 20, a, f2, inj_params['Q'], method='linear')

	elif inj_type == 'wn':
		params_path = Path(git_path + '/shared/injection_params')
		with h5py.File(join(params_path, 'wn_f_low_' + str(f_low) + '_f_high_' + str(f_high) + '_tau_' + str(tau) + '.hdf5'), 'w') as f:
			Hp = np.asarray(f['Hp'])
			Hc = np.asarray(f['Hc'])

		shift = int((t_inj - (wf_times[0] + Tc/2)) * fw)
		Hp = np.roll(Hp, shift)
		Hc = np.roll(Hc, shift)
		# Hp, Hc = white_noise(wf_times, t_inj, inj_params['f_low'], inj_params['f_high'], inj_params['tau'])
	
	hp, hc = gen_waveform(inj_params['A'], alpha, Hp, Hc)
	hp = TimeSeries(hp, t0=wf_times[0], dt=dt)
	hc = TimeSeries(hc, t0=wf_times[0], dt=dt)
	try:
		hp = hp.taper()
		hc = hc.taper()

	except:
		pass

	h_rss = np.sqrt((hp.dot(hp) + hc.dot(hc)) * hp.dt).value
	hp *= inj_params['A'] / h_rss
	hc *= inj_params['A'] / h_rss
	return hp, hc

def load_inject_condition(t_i, t_f, t_inj, ra, dec, pol, alpha, inj_type, inj_params=None, local=False, Tc=16, To=2, fw=2048, 
						  window='tukey', detector='H', qtrans=False, qsplit=False, dT=2.0, hp=None, save=False, data_path=None):
	"""Fucntion to load a chunk, inject a waveform and condition, created to enable parallelizing.
	"""
	vmem = psutil.virtual_memory()
	free_mem = vmem.free >> 20
	avail_mem = vmem.available >> 20
	# if free_mem < 3e5:
	if avail_mem < 3e5:
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

	det_obj = Detector(detector + '1')
	delay = det_obj.time_delay_from_detector(Detector('H1'), ra, dec, t_inj)
	t_inj += delay
	fp, fc = det_obj.antenna_pattern(ra, dec, pol, t_inj)

	wf_times = data.times.value

	hp, hc = gen_inject(wf_times, data.dt, t_inj, alpha, inj_type, inj_params, Tc, fw)
	h = fp * hp + fc * hc
	injected_data = data.inject(h)

	del data
	gc.collect()

	cond_data = condition_data(injected_data, To, fw, window, qtrans, qsplit, dT)

	del injected_data
	gc.collect()

	x = []
	times = []

	for dat in cond_data:
		x.append(dat.values)
		times.append(dat.t0)

	del cond_data
	gc.collect()

	x = np.asarray(x)
	times = np.asarray(times)

	idx = find_closest_index(t_inj, times)

	x = x[idx]
	times = times[idx]

	return x, times

def prep_ccsn(h, sim_times, Tc, fw):
	"""Function to prepare a single polarization of a simulated ccsn waveform - resample, high pass filter and window
	"""
	dt = sim_times[1] - sim_times[0]
	h = TimeSeries(h, t0=sim_times[0], dt=dt)
	h = h.resample(rate=fw, ftype = 'iir', n=20) # downsample to working frequency fw
	h = h.highpass(frequency=11, filtfilt=True) # filter out frequencies below 20Hz
	inj_window = scisig.tukey(M=len(h), alpha=0.08, sym=True)
	h = h * inj_window
	h = h.pad(int((fw * Tc - len(h)) / 2))
	return h


def load_ccsn_wf(ccsn_paper, ccsn_file, Tc, fw):
	"""Function to load a simulated ccsn waveform and prepare it for injection
	"""
	wfs_path = Path(git_path + '/shared/ccsn_wfs/' + ccsn_paper)
	sim_data = [i.strip().split() for i in open(join(wfs_path, ccsn_file)).readlines()]
	if ccsn_paper == 'radice':
		line_s = 1
	else:
		line_s = 0

	# D = D_kpc *  3.086e+21 # cm
	sim_times = np.asarray([float(dat[0]) for dat in sim_data[line_s:]])
	hp = np.asarray([float(dat[1]) for dat in sim_data[line_s:]])
	if ccsn_paper == 'abdikamalov':
		hc = np.zeros(hp.shape)
	else:
		hc = np.asarray([float(dat[2]) for dat in sim_data[line_s:]])

	hp = prep_ccsn(hp, sim_times, Tc, fw)
	hc = prep_ccsn(hc, sim_times, Tc, fw)
	return hp, hc

def load_inject_condition_ccsn(t_i, t_f, t_inj, ra, dec, pol, hp, hc, local=False, Tc=16, To=2, fw=2048, 
							   window='tukey', detector='H', qtrans=False, qsplit=False, dT=2.0, save=False, data_path=None):
	"""Fucntion to load a chunk, inject a waveform and condition, created to enable parallelizing.
	"""
	vmem = psutil.virtual_memory()
	free_mem = vmem.free >> 20
	avail_mem = vmem.available >> 20
	# if free_mem < 3e5:
	if avail_mem < 3e5:
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

	det_obj = Detector(detector + '1')
	delay = det_obj.time_delay_from_detector(Detector('H1'), ra, dec, t_inj)
	t_inj += delay
	fp, fc = det_obj.antenna_pattern(ra, dec, pol, t_inj)

	# wfs_path = Path(git_path + '/shared/ccsn_wfs/' + ccsn_paper)
	# sim_data = [i.strip().split() for i in open(join(wfs_path, ccsn_file)).readlines()]
	# if ccsn_paper == 'radice':
	# 	line_s = 1
	# else:
	# 	line_s = 0

	# D = D_kpc *  3.086e+21 # cm
	# sim_times = np.asarray([float(dat[0]) for dat in sim_data[line_s:]])
	# hp = np.asarray([float(dat[1]) for dat in sim_data[line_s:]]) / D
	# if ccsn_paper == 'abdikamalov':
	# 	hc = np.zeros(hp.shape)
	# else:
	# 	hc = np.asarray([float(dat[2]) for dat in sim_data[line_s:]]) / D

	# dt = sim_times[1] - sim_times[0]
	h = fp * hp + fc * hc
	# h = TimeSeries(h, t0=sim_times[0], dt=dt)

	# h = h.resample(rate=fw, ftype = 'iir', n=20) # downsample to working frequency fw
	# h = h.highpass(frequency=11, filtfilt=True) # filter out frequencies below 20Hz
	# inj_window = scisig.tukey(M=len(h), alpha=0.08, sym=True)
	# h = h * inj_window

	# h = h.pad(int((fw * Tc - len(h)) / 2))

	wf_times = data.times.value

	shift = int((t_inj - (wf_times[0] + Tc/2)) * fw)
	h = np.roll(h.value, shift)
	
	h = TimeSeries(h, t0=wf_times[0], dt=data.dt)
	try:
		h = h.taper()
	except:
		pass

	injected_data = data.inject(h)

	del data
	gc.collect()
	
	cond_data = condition_data(injected_data, To, fw, window, qtrans, qsplit, dT)

	del injected_data
	gc.collect()

	x = []
	times = []

	for dat in cond_data:
		x.append(dat.values)
		times.append(dat.t0)

	del cond_data
	gc.collect()

	x = np.asarray(x)
	times = np.asarray(times)

	idx = find_closest_index(t_inj, times)

	x = x[idx]
	times = times[idx]
	return x, times