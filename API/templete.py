#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.training import moving_averages
import tensorflow as tf
from  API.dataLoad import LoadDatas
from skimage.io import imsave

import os
import argparse
import sys

class Visualize_train(object):
    def __init__(self):
        pass

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def image_summary(self,name,tensor,max_outputs=10):
        tf.summary.image(name, tensor, max_outputs)

    def hist_summary(self,name,values):
        tf.summary.histogram(name, values)

    def scalar_summary(self,name,tensor):
        tf.summary.scalar(name, tensor)

    def merge_all_summary(self):
        return tf.summary.merge_all()

    def FileWriter_summary(self,log_dir,graph=None):
        return tf.summary.FileWriter(log_dir,graph)

class Save_and_load_mode(object):
    def __init__(self,logdir,sess):
        self.saver = tf.train.Saver()
        self.logdir=logdir # 保存模型位置
        self.sess=sess

    def save_model(self,step):
        if not os.path.exists(self.logdir):os.makedirs(self.logdir)
        self.saver.save(self.sess, os.path.join(self.logdir,'model.ckpt'), global_step=step)

    def load_model(self):
        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

class Model(object):
    def __init__(self,X,Y,w,b,learning_rate):#init
        self.X=X
        self.Y=Y
        self.w=w
        self.b=b
        self.learning_rate=learning_rate

    def forward_pred(self,activation='softmax'):
        if activation=='softmax':
            pred=tf.nn.softmax(tf.matmul(self.X, self.w) + self.b)
        else:
            pred=tf.nn.bias_add(tf.matmul(self.X, self.w),self.b)
        return pred

    def loss(self,pred_value,MSE_error=False,one_hot=True):
        if MSE_error:
            return tf.reduce_mean(tf.reduce_sum(tf.square(pred_value-self.Y),reduction_indices=[1]))
        else:
            if one_hot:#softmax
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_value))
            else:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.Y, tf.int32), logits=pred_value))
    def train(self,cross_entropy):
        global_step = tf.Variable(0, trainable=False)
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)
    def evaluate(self,pred_value,one_hot=True):
        if one_hot:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(self.Y, 1))
            # correct_prediction= tf.nn.in_top_k(pred_value, Y, 1)
        else:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.cast(self.y, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5,img_height=28,img_width=28):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

# from IPython.display import Image, display
# Image('images/13_visual_analysis_flowchart.png')

# import time
# start=time.time
# end=time.time
# print end-start

# weights = {
# 'wc1' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
# 'wc2' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
# 'wd1' : tf.Variable(tf.random_normal([(input_height/4)*(input_height/4)* depth_out2,1024])),
# 'out' : tf.Variable(tf.random_normal([1024,n_classes]))
# }
#
# biases = {
# 'bc1' : tf.Variable(tf.random_normal([64])),
# 'bc2' : tf.Variable(tf.random_normal([128])),
# 'bd1' : tf.Variable(tf.random_normal([1024])),
# 'out' : tf.Variable(tf.random_normal([n_classes]))
# }

import matplotlib.pyplot as plt
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm
def plot_image(image):
    # Normalize the image so pixels are between 0.0 and 1.0
    img_norm = normalize_image(image)

    # Plot the image.
    plt.imshow(img_norm, interpolation='nearest')
    plt.show()


def plot_images(images, show_size=100):
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 3)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, 0:show_size, 0:show_size, :]

        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()





