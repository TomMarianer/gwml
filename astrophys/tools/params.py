#!/usr/bin/python
"""
parameters file for conditioning
"""

Tc = 64 # chunk length [s]
To = 2 # chunk overlap [s]
fw = 4 * 4096 # working frequency [Hz]
window = 'tukey' # window used for conditioning
dT = 2 # spectrogram length [s]