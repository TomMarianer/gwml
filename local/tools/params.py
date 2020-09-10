#!/usr/bin/python
"""
parameters file for local
"""

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
