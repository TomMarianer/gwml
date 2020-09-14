#!/usr/bin/python

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal import chirp, iirdesign, filtfilt
from numpy.random import randn
import numpy as np
import pandas as pd
# import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
# from tools_gs_par import *

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
	sig = randn(len(times))
	b, a = iirdesign(wp=(f_low, f_high), ws=(f_low * 2/3., min(f_high * 1.5, fs/2.)), gpass=2, gstop=30, fs=fs)
	Hp = env * filtfilt(b, a, sig)
	Hc = Hp
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

def inject(data, t_inj, inj_type, inj_params):
	"""Inject waveform to data
	"""

	wf_times = data.times.value
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
		Hp, Hc = white_noise(wf_times, t_inj, inj_params['f_low'], inj_params['f_high'], inj_params['tau'])

	hp, hc = gen_waveform(inj_params['A'], inj_params['alpha'], Hp, Hc)
	hp = TimeSeries(hp, t0=wf_times[0], dt=data.dt)
	injected_data = data.inject(hp)
	return injected_data

def load_inject_condition(t_i, t_f, t_inj, inj_type, inj_params=None, local=False, Tc=16, To=2, fw=2048, window='tukey', detector='H', 
						  qtrans=False, qsplit=False, dT=2.0, save=False, data_path=None, hp=None):
	"""Fucntion to load a chunk, inject a waveform and condition, created to enable parallelizing.
	"""
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

	wf_times = data.times.value

	if inj_type == 'ccsn':
		shift = int((t_inj - (wf_times[0] + Tc/2)) * fw)
		hp = np.roll(hp.value, shift)
		
		hp = TimeSeries(hp, t0=times[0], dt=hdata.dt)
		try:
			hp = hp.taper()
		except:
			pass

		injected_data = data.inject(hp)

	else:
		injected_data = inject(data, t_inj, inj_type, inj_params)

	cond_data = condition_data(injected_data, To, fw, window, qtrans, qsplit, dT)

	x = []
	times = []

	for dat in cond_data:
		x.append(dat.values)
		times.append(dat.t0)

	x = np.asarray(x)
	times = np.asarray(times)

	idx = find_closest_index(t_inj, times)

	x = x[idx]
	times = times[idx]
	return x, times