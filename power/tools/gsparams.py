#!/usr/bin/python
"""
parameters file for cnn
"""

extract_model = 'resnet152v2' # pre-trained model to use
input_size = (299, 299) # input size for the network's first layer
train_set = 'fromraw_gs_wrap_no_ty' # training set the network is trained on
opt_method = 'adadelta' # optimization method used
model_num = '15'
model_name = 'gpu_test_' + model_num

