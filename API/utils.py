#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import  #绝对import，优先使用系统自带的包
from __future__ import division
from __future__ import print_function  #优先使用新的发行版特性
import numpy as np
import os, sys, time
import tensorflow as tf
import scipy
from API.convOpt import *

# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = create_var("beta", [num_inputs,],
                               initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs,],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_var("moving_mean", [num_inputs,],
                                 initializer=tf.zeros_initializer(),
                                 trainable=False)
        moving_variance = create_var("moving_variance", [num_inputs],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

# https://blog.csdn.net/huachao1001/article/details/78501928
# 而只保存一部分变量，可以通过指定variables/collections。在创建tf.train.Saver实例时，通过将需要保存的变量构造list或者dictionary，传入到Saver中
# saver = tf.train.Saver([w1,w2])
# saver.save(sess, './checkpoint_dir/MyModel',global_step=step,write_meta_graph=False)
# tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
# 如果我们不对tf.train.Saver指定任何参数，默认会保存所有变量。如果你不想保存所有变量，
# # MyModel.meta文件保存的是图结构，meta文件是pb（protocol buffer）格式文件，包含变量、op、集合等。
# # ckpt文件是二进制文件，保存了所有的weights、biases、gradients等变量。在tensorflow 0.11之前，保存在.ckpt文件中。0.11后，通过两个文件保存,如：
# # MyModel.data-00000-of-00001
# # MyModel.index
# # 我们还可以看，checkpoint_dir目录下还有checkpoint文件，该文件是个文本文件，里面记录了保存的最新的checkpoint文件以及其它checkpoint文件列表。在inference时，可以通过修改这个文件，指定使用哪个model
#
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
#     print(sess.run('w1:0'))
#
# import tensorflow as tf
#
# sess = tf.Session()
# # 先加载图和变量
# saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./'))
#
# # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("w1:0")
# w2 = graph.get_tensor_by_name("w2:0")
# feed_dict = {w1: 13.0, w2: 17.0}
#
# #接下来，访问你想要执行的op
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#
# # 在当前图中能够加入op
# add_on_op = tf.multiply(op_to_restore, 2)
#
# print (sess.run(add_on_op, feed_dict))
# # 打印120.0==>(13+17)*2*2
#
# 如果只想恢复图的一部分，并且再加入其它的op用于fine-tuning。只需通过graph.get_tensor_by_name()方法获取需要的op，并且在此基础上建立图，看一个简单例子，假设我们需要在训练好的VGG网络使用图，并且修改最后一层，将输出改为2，用于fine-tuning新数据：
#
# ......
# ......
# saver = tf.train.import_meta_graph('vgg.meta')
# # 访问图
# graph = tf.get_default_graph()
#
# #访问用于fine-tuning的output
# fc7= graph.get_tensor_by_name('fc7:0')
#
# #如果你想修改最后一层梯度，需要如下
# fc7 = tf.stop_gradient(fc7) # It's an identity function
# fc7_shape= fc7.get_shape().as_list()
#
# new_outputs=2
# weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
# biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
# output = tf.matmul(fc7, weights) + biases
# pred = tf.nn.softmax(output)
#
# # Now, you run this with fine-tuning data in sess.run()

# # x = tf.nn.bias_add(x,b)
#
# %matplotlib inline
# tf.InteractiveSession()
# tf.reduce_sum(b,reduction_indices = 1).eval()
# a.get_shape()
# tf.reshape(a,(1,4)).eval()
# print(sess.run([mul_x_y, final_op]))
# a = np.ones((3,3))
# b = tf.convert_to_tensor(a)
#
# cost = tf.reduce_mean(( (y_ * tf.log(pred)) +
#         ((1 - y_) * tf.log(1.0 - pred)) ) * -1)
#
# var_1 = tf.Variable(0,name='var_1')
# add_op = tf.add(var_1,tf.constant(1))
# upd_op = tf.assign(var_1,add_op)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in xrange(5):
#         print(sess.run(upd_op))
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# writer = tf.summary.FileWriter("./Downloads/XOR1_logs", sess.graph_def)
# sess.run(init)
# for i in range(100000):
#         sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
#         if i % 10000 == 0:
#             print('Epoch ', i)
#             print('Prediction:', sess.run(pred,feed_dict={x_: XOR_X, y_: XOR_Y}))
#             print('Weights from input to hidden layer:', sess.run(w1))
#             print('Bias in the hidden layer:', sess.run(b1))
#             print('Weights from hidden layer to output layer:', sess.run(w2))
#             print('Bias in the output layer:', sess.run(b2))
#             print('Cost:', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
# #----------------------------------------------------------------------------------------------
# print('Final Prediction', sess.run(pred, feed_dict={x_: XOR_X, y_: XOR_Y}))
#
# learning_rate = 0.01
# num_epochs = 1000
# cost_trace = []
# pred = tf.matmul(X,w)
# error = pred - Y
# cost = tf.reduce_mean(tf.square(error))
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# sess.run(train_op, feed_dict={X: X_input, Y: Y_input})
# cost_trace.append(sess.run(cost, feed_dict={X: X_input, Y: Y_input}))
# plt.plot(cost_trace#fig1
# fig, ax = plt.subplots()#fig2
# plt.scatter(Y_input,pred_)
# ax.set_xlabel('Actual House price')
# ax.set_ylabel('Predicted House price')
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))
# accuracy_ = np.mean(np.argmax(sess.run(out, feed_dict={X: train_X, Y: train_Y}), axis=1) == np.argmax(train_Y, axis=1))loss_trace.append(loss_)
# print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss),"Training accuracy","{:.5f}".format(acc))
#
# import matplotlib.pyplot as plt
# %matplotlib inline
# f, a = plt.subplots(1, 10, figsize=(10, 2))
# print 'Actual digits:   ', np.argmax(test_Y[0:10],axis=1)
# print 'Predicted digits:',test_pred[0:10]
# print 'Actual images of the digits follow:'
# for i in range(10):
#         a[i].imshow(np.reshape(test_X[i],(28, 28)))
#
# CNN ref pro
# for i in xrange(1,1+row):
#     for j in xrange(1,1+col):
#         arr_chunk = np.zeros((3,3))
#         for k,k1 in zip(xrange(i-1,i+2),xrange(3)):
#             for l,l1 in zip(xrange(j-1,j+2),xrange(3)):
#                 arr_chunk[k1,l1] = image1[k,l]
#         image_out[i-1,j-1] = np.sum(np.multiply(arr_chunk,filter_kernel_flipped))
# print "2D convolution implementation"
#
# import numpy as np
# import matplotlib.pyplot as plt# Take a 7x7 image as example
# image = np.array([[1, 2, 3, 4, 5, 6, 7],
#                  [8, 9, 10, 11, 12, 13, 14],
#                  [15, 16, 17, 18, 19, 20, 21],
#                  [22, 23, 24, 25, 26, 27, 28],
#                  [29, 30, 31, 32, 33, 34, 35],
#                  [36, 37, 38, 39, 40, 41, 42],
#                  [43, 44, 45, 46, 47, 48, 255]])
# plt.imshow(image)
#
# from layers import conv_layer, max_pool_2x2, full_layer
# (layer.py)
#
# g1 = tf.get_default_graph()
# g2 = tf.Graph()
# print(g1 is tf.get_default_graph())
# with g2.as_default():
#     print(g1 is tf.get_default_graph())
# print(g1 is tf.get_default_graph())
# t,f,t
#
# titles = ['Normal','Truncated Normal','Uniform']
#     ax.set_xlabel('Values',fontsize=20)
# axarr[0].set_ylabel('Frequency',fontsize=20)
# plt.suptitle('Initialized values',fontsize=30, y=1.15)
# plt.tight_layout()
# plt.savefig('histograms.png', bbox_inches='tight', format='png', dpi=200, pad_inches=0,transparent=True)
# plt.show()
#
# with tf.Graph().as_default():
#     c1 = tf.constant(4,dtype=tf.float64,name='c')
#     with tf.name_scope("prefix_name"):
#         c2 = tf.constant(4,dtype=tf.int32,name='c')
#         c3 = tf.constant(4,dtype=tf.float64,name='c')
# print(c1.name)
# print(c2.name)
# print(c3.name)
# c:0
# prefix_name/c:0
# prefix_name/c_1:0
#
#
# import matplotlib.pyplot as plt
# % matplotlib inline
# sess = tf.InteractiveSession()
# # === Noramal and Truncated normal distributions ===
# mean = 0
# std = 1
# x_normal = tf.random_normal((1,50000),mean,std).eval()
# x_truncated = tf.truncated_normal((1,50000),mean,std).eval()
# # === Uniform distribution
# minval = -2
# maxval = 2
# x_uniform = tf.random_uniform((1,50000),minval,maxval).eval()
# sess.close()
# def simpleaxis(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
# #     ax.set_ylim([-1.1,1.1])
#     ax.tick_params(axis='both', which='major', labelsize=15)
# def get_axis_limits(ax, scale=.8):
#     return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale
# f,axarr = plt.subplots(1,3,figsize=[15,4],sharey=True)
# titles = ['Normal','Truncated Normal','Uniform']
# print(x_normal.shape)
# for i,x in enumerate([x_normal,x_truncated,x_uniform]):
#     ax = axarr[i]
#     ax.hist(x[0],bins=100,color='b',alpha=0.4)
#     ax.set_title(titles[i],fontsize=20)
#     ax.set_xlabel('Values',fontsize=20)
#     ax.set_xlim([-5,5])
#     ax.set_ylim([0,1800])
#     simpleaxis(ax)
# axarr[0].set_ylabel('Frequency',fontsize=20)
# plt.suptitle('Initialized values',fontsize=30, y=1.15)
# for ax,letter in zip(axarr,['A','B','C']):
#     simpleaxis(ax)
#     ax.annotate(letter, xy=get_axis_limits(ax),fontsize=35)
# plt.tight_layout()
# plt.savefig('histograms.png', bbox_inches='tight', format='png', dpi=200, pad_inches=0,transparent=True)
# plt.show()

# def plot_images(images, cls_true, cls_pred=None):
#     assert len(images) == len(cls_true) == 9
#     # Create figure with 3x3 sub-plots.
#     fig, axes = plt.subplots(3, 3)
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     for i, ax in enumerate(axes.flat):
#         # Plot image.
#         ax.imshow(images[i].reshape(img_shape), cmap='binary')
#         # Show true and predicted classes.
#         if cls_pred is None:
#             xlabel = "True: {0}".format(cls_true[i])
#         else:
#             xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
#         ax.set_xlabel(xlabel)
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
# # Get the first images from the test-set.
# images = data.test.images[0:9]
# # Get the true classes for those images.
# cls_true = data.test.cls[0:9]
# # Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)
#
#     cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
#     # Print the confusion matrix as text.
#     print(cm)
#     # Plot the confusion matrix as an image.
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.tight_layout()
#     plt.colorbar()
#     tick_marks = np.arange(num_classes)
#     plt.xticks(tick_marks, range(num_classes))
#     plt.yticks(tick_marks, range(num_classes))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#
# def plot_weights():
#     # Get the values for the weights from the TensorFlow variable.
#     w = session.run(weights)
#     # Get the lowest and highest values for the weights.
#     # This is used to correct the colour intensity across
#     # the images so they can be compared with each other.
#     w_min = np.min(w)
#     w_max = np.max(w)
#     # Create figure with 3x4 sub-plots,
#     # where the last 2 sub-plots are unused.
#     fig, axes = plt.subplots(3, 4)
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     for i, ax in enumerate(axes.flat):
#         # Only use the weights for the first 10 sub-plots.
#         if i<10:
#             # Get the weights for the i'th digit and reshape it.
#             # Note that w.shape == (img_size_flat, 10)
#             image = w[:, i].reshape(img_shape)
#             # Set the label for the sub-plot.
#             ax.set_xlabel("Weights: {0}".format(i))
#             # Plot the image.
#             ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
#         # Remove ticks from each sub-plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#
# def plot_image(image):
#     plt.imshow(image.reshape(img_shape),
#                interpolation='nearest',
#                cmap='binary')
#     plt.show()
#
# def plot_conv_weights(weights, input_channel=0):
#     # Assume weights are TensorFlow ops for 4-dim variables
#     # e.g. weights_conv1 or weights_conv2.
#     # Retrieve the values of the weight-variables from TensorFlow.
#     # A feed-dict is not necessary because nothing is calculated.
#     w = session.run(weights)
#     # Get the lowest and highest values for the weights.
#     # This is used to correct the colour intensity across
#     # the images so they can be compared with each other.
#     w_min = np.min(w)
#     w_max = np.max(w)
#     # Number of filters used in the conv. layer.
#     num_filters = w.shape[3]
#     # Number of grids to plot.
#     # Rounded-up, square-root of the number of filters.
#     num_grids = math.ceil(math.sqrt(num_filters))
#     # Create figure with a grid of sub-plots.
#     fig, axes = plt.subplots(num_grids, num_grids)
#     # Plot all the filter-weights.
#     for i, ax in enumerate(axes.flat):
#         # Only plot the valid filter-weights.
#         if i<num_filters:
#             # Get the weights for the i'th filter of the input channel.
#             # See new_conv_layer() for details on the format
#             # of this 4-dim tensor.
#             img = w[:, :, input_channel, i]
#             # Plot image.
#             ax.imshow(img, vmin=w_min, vmax=w_max,
#                       interpolation='nearest', cmap='seismic')
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()
#
# def plot_conv_layer(layer, image):
#     # Assume layer is a TensorFlow op that outputs a 4-dim tensor
#     # which is the output of a convolutional layer,
#     # e.g. layer_conv1 or layer_conv2.
#     # Create a feed-dict containing just one image.
#     # Note that we don't need to feed y_true because it is
#     # not used in this calculation.
#     feed_dict = {x: [image]}
#     # Calculate and retrieve the output values of the layer
#     # when inputting that image.
#     values = session.run(layer, feed_dict=feed_dict)
#     # Number of filters used in the conv. layer.
#     num_filters = values.shape[3]
#     # Number of grids to plot.
#     # Rounded-up, square-root of the number of filters.
#     num_grids = math.ceil(math.sqrt(num_filters))
#     # Create figure with a grid of sub-plots.
#     fig, axes = plt.subplots(num_grids, num_grids)
#     # Plot the output images of all the filters.
#     for i, ax in enumerate(axes.flat):
#         # Only plot the images for valid filters.
#         if i<num_filters:
#             # Get the output image of using the i'th filter.
#             # See new_conv_layer() for details on the format
#             # of this 4-dim tensor.
#             img = values[0, :, :, i]
#             # Plot image.
#             ax.imshow(img, interpolation='nearest', cmap='binary')
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()

# TensorFlow-Tutorials 02
# def plot_conv_layer(layer, image):
#     # Assume layer is a TensorFlow op that outputs a 4-dim tensor
#     # which is the output of a convolutional layer,
#     # e.g. layer_conv1 or layer_conv2.
#     # Create a feed-dict containing just one image.
#     # Note that we don't need to feed y_true because it is
#     # not used in this calculation.
#     feed_dict = {x: [image]}
#     # Calculate and retrieve the output values of the layer
#     # when inputting that image.
#     values = session.run(layer, feed_dict=feed_dict)
#     # Number of filters used in the conv. layer.
#     num_filters = values.shape[3]
#     # Number of grids to plot.
#     # Rounded-up, square-root of the number of filters.
#     num_grids = math.ceil(math.sqrt(num_filters))
#     # Create figure with a grid of sub-plots.
#     fig, axes = plt.subplots(num_grids, num_grids)
#     # Plot the output images of all the filters.
#     for i, ax in enumerate(axes.flat):
#         # Only plot the images for valid filters.
#         if i < num_filters:
#             # Get the output image of using the i'th filter.
#             # See new_conv_layer() for details on the format
#             # of this 4-dim tensor.
#             img = values[0, :, :, i]
#             # Plot image.
#             ax.imshow(img, interpolation='nearest', cmap='binary')
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()
#
# TensorFlow-Tutorials 03_PrettyTensor
# prettytensor
# with pt.defaults_scope(activation_fn=tf.nn.relu):
#     y_pred, loss = x_pretty.\
#         conv2d(kernel=5, depth=16, name='layer_conv1').\
#         max_pool(kernel=2, stride=2).\
#         conv2d(kernel=5, depth=36, name='layer_conv2').\
#         max_pool(kernel=2, stride=2).\
#         flatten().\
#         fully_connected(size=128, name='layer_fc1').\
#         softmax_classifier(num_classes=num_classes, labels=y_true)



# print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(loss), "Training accuracy", "{:.5f}".format(acc))


    # f, a = plt.subplots(1, 10, figsize=(10, 2))
    # for i in range(10):
    #     a[i].imshow(np.reshape(mnist.test.images[i],(28, 28)))

# 代替with
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

