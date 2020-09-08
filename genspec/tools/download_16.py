#!/usr/bin/env python3
# script for downloading bulk ligo data

# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import sys
from pathlib import Path
from os import listdir, makedirs
from os.path import isfile, join, exists

pyversion = sys.version_info.major
if pyversion == 2: 
	import urllib2
else:
	import urllib.request

# -- Handy function to download data file, and return the filename
def download(url, detector):
	data_path = Path('/arch/tommaria/data/bulk_data/16KHZ/' + detector)
	filename = join(data_path, url.split('/')[-1])
	files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
	if filename in files: # If file already downloaded, don't download again and exit
		print('File already downloaded')
		return

	if pyversion == 2: 
		r = urllib2.urlopen(url).read()
		f = open(filename, 'w')   # write it to the right filename
		f.write(r)
		f.close()
	else:
		urllib.request.urlretrieve(url, filename)  
	#print("File download complete")

data_path = Path('../bulk_data_urls')
urls = dict()
for detector in ['H1', 'L1']:
	filename = 'O1_16KHZ' + '_' + detector + '_' + 'files'
	f = open(join(data_path, filename), 'r')
	urls[detector] = f.read().split('\n')[:-1]

for detector in ['H1', 'L1']:
	filename = 'O2_16KHZ_R1' + '_' + detector + '_' + 'files'
	f = open(join(data_path, filename), 'r')
	urls[detector].extend(f.read().split('\n')[:-1])

print(len(urls['H1']))
print(len(urls['L1']))

# don't really need this since added the file downloaded check in the download function
# files_with_glitch = []
# with open('files_with_glitch.txt') as f:
# 	for line in f:
# 		files_with_glitch.append(line.rstrip('\n'))

detector = 'L1' # choose the detector whose files you want to download

urls2download = urls[detector][1678+1715:1679+1718] # indices of the files you want to download
# print(urls2download)
# list of the file indices already downloaded for each detector
# H - 0:2500,2607:2613,2613:2614,3186:3199,3544:3548,3751:3755,4150:4161,5024:5028,5071:5075,5248:5252,5334:5343,5390:5394,5403:5406,5503:5521
# L - 0:1500,1500:2000,2000:2500,1916:1926,2175:2177,2321:2327,2853:2864,3393:3398,3754:3765,4922:4926,4969:4973,5169:5173,5263:5272,5335:5337,5322:5328,5441:5459

import time
start_time = time.time()

for url in urls2download:
	print(url)
	# if url in files_with_glitch:
	#     continue
	download(url, detector)

print("--- Execution time is %.7s minutess ---\n" % ((time.time() - start_time)/60))
