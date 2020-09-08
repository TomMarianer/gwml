#!/usr/bin/env python3
# script for downloading files containing gravity spy glitches
# there are differences from the download_16.py script that result from when I wrote this
# they should be fixed some day

# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import sys
from os import listdir, makedirs

pyversion = sys.version_info.major
if pyversion == 2: 
	import urllib2
else:
	import urllib.request

# -- Handy function to download data file, and return the filename
def download(url):
	filename = '/storage/fast/users/tommaria/data/bulk_data/16KHZ/gravityspy/' + url.split('/')[-1]
	print('Downloading ' + url )
	if pyversion == 2: 
		r = urllib2.urlopen(url).read()
		f = open(filename, 'w')   # write it to the right filename
		f.write(r)
		f.close()
	else:
		urllib.request.urlretrieve(url, filename)  
	print("File download complete")

urls = []

with open('files_with_glitch.txt') as f:
	for line in f:
		urls.append(line.rstrip('\n'))

urls2download = urls[500:1500]

dirs = listdir('/storage/fast/users/tommaria/data/bulk_data/16KHZ/gravityspy')
for url in urls2download:
	if url.split('/')[-1] not in dirs:
		download(url)
	else:
		print('URL ' + url + 'exists')
print('Done')
