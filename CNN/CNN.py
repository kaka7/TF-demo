#encoding=utf-8
from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
from API.convOpt import conv_layer, max_pool_2x2, full_layer
from  API.dataLoad import LoadDatas
from API.templete import Visualize_train
FLAGS = None
DATA_DIR = '/home/naruto/PycharmProjects/data/mnist'
MINIBATCH_SIZE = 64
STEPS = 500

def tf_model_build():
    # Visualize_train = Visualize_train()
    # Input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28 * 28 * 1], 'x')
        y_ = tf.placeholder(tf.float32, [None, 10], 'y_')
        with tf.name_scope('input_reshape'):
             image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
             Visualize_train.image_summary('x', image_shaped_input, 10)
        input_model=LoadDatas(FLAGS.data_dir,FLAGS.batch_size_num,one_hot=FLAGS.one_hot)
        # weight init
        with tf.name_scope('line_layer'):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.random_normal([28*28*1, 10])) # 二分类
                Visualize_train.variable_summaries(weights)
            with tf.name_scope('biases'):
                bias = tf.Variable(tf.random_normal([10]))
                Visualize_train.variable_summaries(bias)


        model=Model_interface(x,y_,weights,bias,FLAGS.learning_rate)
        with tf.name_scope('Wx_plus_b'):
            y=model.forward_pred(classifier='softmax')
            Visualize_train.hist_summary('pred',y)

        with tf.name_scope('total_loss'):
            cross_entropy=model.loss(y,MSE_flag=False,one_hot=FLAGS.one_hot)
            Visualize_train.scalar_summary('cross_entropy', cross_entropy)

        train_op=model.train(cross_entropy)

        with tf.name_scope('accuracy'):
            accuracy=model.evaluate(y,one_hot=FLAGS.one_hot)
            Visualize_train.scalar_summary('accuracy', accuracy)

        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  #
    conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf初始化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(STEPS):
            batch = mnist.train.next_batch(MINIBATCH_SIZE)
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            if i % 100 == 0:
                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean(
            [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])
    print("test accuracy: {}".format(test_accuracy))

if __name__=="__main__":
    mnist = LoadDatas(DATA_DIR, MINIBATCH_SIZE).mnistDataSet
    tf_model_build()







