#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import  #绝对import，优先使用系统自带的包
from __future__ import division
from __future__ import print_function  #优先使用新的发行版特性
import numpy as np
import os, sys, time
import tensorflow as tf

from sklearn.datasets import load_boston

def read_infile():
    data = load_boston()
    features = np.array(data.data)
    target = np.array(data.target)
    return features,target

#----------------------------------------------------------------------------------------------

# Normalize the features by Z scaling i.e. subract form each feature value its mean and then divide by its
# standard deviation. Accelerates Gradient Descent.

#----------------------------------------------------------------------------------------------

def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
def append_bias(features,target):
    n_samples = features.shape[0]
    n_features = features.shape[1]
    intercept_feature  = np.ones((n_samples,1))
    X = np.concatenate((features,intercept_feature),axis=1)
    X = np.reshape(X,[n_samples,n_features +1])
    Y = np.reshape(target,[n_samples,1])
    return X,Y

#----------------------------------------------------------------------------------------------

# Execute the functions to read, normalize and add append bias term to the data

#----------------------------------------------------------------------------------------------

features,target = read_infile()
z_features = feature_normalize(features)
X_input,Y_input = append_bias(z_features,target)
num_features = X_input.shape[1]

