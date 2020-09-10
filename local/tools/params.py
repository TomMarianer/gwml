#!/usr/bin/python
"""
parameters file for local
"""

from pathlib import Path

### model parameters

condition_method = 'gs_wrap_no_ty'
model = 'new/resnet152v2/adadelta/gpu_test_15'

### kde parameters

# map grid ranges
x_range = (-16, 22) 
y_range = (-25, 21.5)

# estimator parameters
kernel = 'gaussian'
bandwidth = 0.3

### thresholds

kde_th = -11.5
gram_th = 5100

### paths

features_path = Path('/Users/tommarianer/LOSC Data/gravityspy/features')
conditioned_path = Path('/Users/tommarianer/LOSC Data/conditioned_data/16KHZ')