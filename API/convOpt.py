#encoding=utf-8

import tensorflow as tf
from tensorflow.python.training import moving_averages

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    # return tf.Variable(tf.random_normal(shape,stddev=0.01))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# max pool layer
def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                              [1, stride, stride, 1], padding="SAME")

# avg pool layer
def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")


# create weight variable
def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)
def con2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        kernel = create_var("kernel", [kernel_size, kernel_size,num_inputs, num_outputs],conv2d_initializer())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1],padding="SAME")

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#shape[3]
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def fc(x, num_outputs, scope="fc"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        weight = create_var("weight", [num_inputs, num_outputs],
                            fc_initializer())
        bias = create_var("bias", [num_outputs,],
                          tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, weight, bias)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b




conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d
fc_initializer = tf.contrib.layers.xavier_initializer









