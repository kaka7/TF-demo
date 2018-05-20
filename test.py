#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import  #绝对import，优先使用系统自带的包
import numpy as np
import os, sys, time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.examples.tutorials.c
# from tf.
tf.nn.conv2d()
import numpy as np
#os.chdir("" )
print("当前工作目录 : %s" % os.getcwd())
# mnist=input_data.read_data_sets('/home/naruto/PycharmProjects/data/mnist',one_hot='True')
# batch=mnist.train.next_batch(1)
# print type(batch)
# print np.shape(np.asarray(batch))

# print tf.dtypes(mnist.train.next_batch(12)).eval()
# import matplotlib.pyplot as plt
# plt.imshow(mnist.train.images[0].reshape([28,28]))
# # f, a = plt.subplots(1, 10, figsize=(10, 2))
# # for i in range(10):
# #     a[i].imshow(np.reshape(mnist.test.images[i], (28, 28)))

# import cv2
# import matplotlib.pyplot as plt
# from scipy.signal import convolve2d
#
# img = cv2.imread('/home/naruto/PycharmProjects/data/marathon.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap='gray')
# mean = 0
# var = 100
# sigma = var**0.5
# row,col = np.shape(gray)
# gauss = np.random.normal(mean,sigma,(row,col))
# gauss = gauss.reshape(row,col)
# gray_noisy = gray + gauss
# print "Image after applying Gaussian Noise"
# plt.imshow(gray_noisy,cmap='gray')

mnist=input_data.read_data_sets("/home/naruto/PycharmProjects/data/mnist/")
print(set(mnist.train.images[0]))
print(np.shape(mnist.train.images[0]))
# from tensorflow.python.keras.models import
import tensorflow as tf
tf.WholeFileReader
tf.__version__