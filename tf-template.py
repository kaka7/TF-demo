#encoding=utf-8
# import load
# from dp import Network
# 一 创建图
# 定义变量
# tf.constant()
# tf.Variable()
# 二 创建会话
# sess=tf.Session
# 计算
# sess.run()

#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

"""说明
数据：mnist
模型建立 Model
数据的输入 Inputs
模型保存与提取 Save_and_load_mode
模型可视化 TensorBoard
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import sys



class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction

    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
import functools

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator






class ResNet50(object):
    def __init__(self, inputs, num_classes=1000, is_training=True,
                 scope="resnet50"):
        self.inputs =inputs
        self.is_training = is_training
        self.num_classes = num_classes

        with tf.variable_scope(scope):
            # construct the model
            net = conv2d(inputs, 64, 7, 2, scope="conv1") # -> [batch, 112, 112, 64]
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            net = max_pool(net, 3, 2, scope="maxpool1")  # -> [batch, 56, 56, 64]
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,
                              scope="block2")           # -> [batch, 56, 56, 256]
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")
                                                        # -> [batch, 28, 28, 512]
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")
                                                        # -> [batch, 14, 14, 1024]
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")
                                                        # -> [batch, 7, 7, 2048]
            net = avg_pool(net, 7, scope="avgpool5")    # -> [batch, 1, 1, 2048]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze") # -> [batch, 2048]
            self.logits = fc(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            self.predictions = tf.nn.softmax(self.logits)


    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride,
                                   is_training=is_training, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, is_training=is_training,
                                       scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = conv2d(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)


if __name__=="__main__":
    # 设置必要参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1000,
                        help = 'Number of steps to run trainer.')
    parser.add_argument('--disp_step', type=int, default=100,
                        help='Number of steps to display.')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of mini training samples.')
    parser.add_argument('--one_hot', type=bool, default=True,
                        help='One-Hot Encoding.')
    parser.add_argument('--data_dir', type=str, default='./MNIST_data/',
            help = 'Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 启动TensorBoard: tensorboard --logdir=path/to/log-directory
# tensorboard --logdir='log_dir'

from sklearn.datasets import load_boston
def read_infile():
    data = load_boston()
    features = np.array(data.data)
    target = np.array(data.target)
    return features,target
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
# error = pred - Y
# cost = tf.reduce_mean(tf.square(error))
# cost_trace = []
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.plot(cost_trace)
# fig, ax = plt.subplots()
# plt.scatter(Y_input,pred_)
# ax.set_xlabel('Actual House price')
# ax.set_ylabel('Predicted House price')
# cost_trace.append(sess.run(cost,feed_dict={X:X_input,Y:Y_input}))
def read_infile():
    mnist = input_data.read_data_sets("/home/naruto/PycharmProjects/data/mnist", one_hot=True)
    train_X, train_Y, test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return train_X, train_Y, test_X, test_Y
def weights_biases_placeholder(n_dim, n_classes):
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    w = tf.Variable(tf.random_normal([n_dim, n_classes], stddev=0.01), name='weights')
    b = tf.Variable(tf.random_normal([n_classes]), name='weights')
    return X, Y, w, b
def forward_pass(w, b, X):
    out = tf.matmul(X, w) + b
    return out
def multiclass_cost(out, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))
    return cost
def init():
    return tf.global_variables_initializer()
def train_op(learning_rate, cost):
    op_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return op_train
train_X, train_Y, test_X, test_Y = read_infile()
X, Y, w, b = weights_biases_placeholder(train_X.shape[1], train_Y.shape[1])
out = forward_pass(w, b, X)
cost = multiclass_cost(out, Y)
learning_rate, epochs = 0.01, 1000
op_train = train_op(learning_rate, cost)
init = init()
loss_trace = []
accuracy_trace = []
with tf.Session() as sess:
    sess.run(init)

    for i in xrange(epochs):
        sess.run(op_train, feed_dict={X: train_X, Y: train_Y})
        loss_ = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        accuracy_ = np.mean(
            np.argmax(sess.run(out, feed_dict={X: train_X, Y: train_Y}), axis=1) == np.argmax(train_Y, axis=1))
        loss_trace.append(loss_)
        accuracy_trace.append(accuracy_)
        if (((i + 1) >= 100) and ((i + 1) % 100 == 0)):
            print
            'Epoch:', (i + 1), 'loss:', loss_, 'accuracy:', accuracy_

    print
    'Final training result:', 'loss:', loss_, 'accuracy:', accuracy_
    loss_test = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
    test_pred = np.argmax(sess.run(out, feed_dict={X: test_X, Y: test_Y}), axis=1)
    accuracy_test = np.mean(test_pred == np.argmax(test_Y, axis=1))
    print
    'Results on test dataset:', 'loss:', loss_test, 'accuracy:', accuracy_test
import matplotlib.pyplot as plt
# %matplotlib inline
f, a = plt.subplots(1, 10, figsize=(10, 2))
print('Actual digits:   ', np.argmax(test_Y[0:10],axis=1))
print('Predicted digits:',test_pred[0:10])
print('Actual images of the digits follow:')
for i in range(10):
        a[i].imshow(np.reshape(test_X[i],(28, 28)))


def read_infile():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_X, train_Y, test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return train_X, train_Y, test_X, test_Y


def weights_biases_placeholder(n_dim, n_classes):
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    w = tf.Variable(tf.random_normal([n_dim, n_classes], stddev=0.01), name='weights')
    b = tf.Variable(tf.random_normal([n_classes]), name='weights')
    return X, Y, w, b


def forward_pass(w, b, X):
    out = tf.matmul(X, w) + b
    return out


def multiclass_cost(out, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))
    return cost


def init():
    return tf.global_variables_initializer()


def train_op(learning_rate, cost):
    op_train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return op_train


train_X, train_Y, test_X, test_Y = read_infile()
X, Y, w, b = weights_biases_placeholder(train_X.shape[1], train_Y.shape[1])
out = forward_pass(w, b, X)
cost = multiclass_cost(out, Y)
learning_rate, epochs, batch_size = 0.01, 1000, 1000
num_batches = train_X.shape[0] / batch_size
op_train = train_op(learning_rate, cost)
init = init()
epoch_cost_trace = []
epoch_accuracy_trace = []

with tf.Session() as sess:
    sess.run(init)

    for i in xrange(epochs):
        epoch_cost, epoch_accuracy = 0, 0

        for j in xrange(num_batches):
            sess.run(op_train, feed_dict={X: train_X[j * batch_size:(j + 1) * batch_size],
                                          Y: train_Y[j * batch_size:(j + 1) * batch_size]})
            actual_batch_size = train_X[j * batch_size:(j + 1) * batch_size].shape[0]
            epoch_cost += actual_batch_size * sess.run(cost, feed_dict={X: train_X[j * batch_size:(j + 1) * batch_size],
                                                                        Y: train_Y[
                                                                           j * batch_size:(j + 1) * batch_size]})

        epoch_cost = epoch_cost / float(train_X.shape[0])
        epoch_accuracy = np.mean(
            np.argmax(sess.run(out, feed_dict={X: train_X, Y: train_Y}), axis=1) == np.argmax(train_Y, axis=1))
        epoch_cost_trace.append(epoch_cost)
        epoch_accuracy_trace.append(epoch_accuracy)

        if (((i + 1) >= 100) and ((i + 1) % 100 == 0)):
            print
            'Epoch:', (i + 1), 'Average loss:', epoch_cost, 'accuracy:', epoch_accuracy

    print
    'Final epoch training results:', 'Average loss:', epoch_cost, 'accuracy:', epoch_accuracy
    loss_test = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
    test_pred = np.argmax(sess.run(out, feed_dict={X: test_X, Y: test_Y}), axis=1)
    accuracy_test = np.mean(test_pred == np.argmax(test_Y, axis=1))
    print
    'Results on test dataset:', 'Average loss:', loss_test, 'accuracy:', accuracy_test

# A sheet of Tensorflow snippets/tips
#### Fancy indexing

tf.gather_nd(params, indices)
retrieves slices from tensor ```params``` by integer tensor ```indices```, similar to Numpy's indexing. When confused, recall this single rule: **only the last dimension of ```indices``` slices ```params```, then that dimension is "replaced" with the slices**. Then we can ses:
  * ```indices.shape[-1] <= rank(params)```: The last dimension of ```indices``` must be no greater than the rank of ```params```, otherwise it can't slice.
  * Result tensor shape is ```indices.shape[:-1] + params.shape[indices.shape[-1]:]```, example:
  ```python
  # params has shape [4, 5, 6].
  params = tf.reshape(tf.range(0, 120), [4, 5, 6])
  # indices has shape [3, 2].
  indices = tf.constant([[2, 3], [0, 1], [1, 2]], dtype=tf.int32)
  # slices has shape [3, 6].
  slices = tf.gather_nd(params, indices)
  ```

#### Don't forget to reset default graph in Jupyter notebook:
If you forgot to reset default Tensorflow graph (or create a new graph) in a Jupyter notebook cell, and run that cell for a few times then you may get weird results.

#### Watch out! ```tf.where``` can spawn NaN in gradients:
If either branch in ```tf.where``` contains Inf/NaN then it produces NaN in gradients, e.g.:
```python
log_s = tf.constant([-100., 100.], dtype=tf.float32)
# Computes 1.0 / exp(log_s), in a numerically robust way.
inv_s = tf.where(log_s >= 0.,
                 tf.exp(-log_s),  # Creates Inf when -log_s is large.
                 1. / (tf.exp(log_s) + 1e-6))  # tf.exp(log_s) is Inf with large log_s.
grad_log_s = tf.gradients(inv_s, [log_s])
with tf.Session() as sess:
    inv_s, grad_log_s = sess.run([inv_s, grad_log_s])
    print(inv_s)  # [  1.00000000e+06   3.78350585e-44]
    print(grad_log_s)  # [array([ nan,  nan], dtype=float32)]
```

#### Shapes:
- ```tensor.shape``` returns tensor's static shape, while the graph is being built.
- ```tensor.shape.as_list()``` returns the static shape as a integer list.
- ```tensor.shape[i].value``` returns the static shape's i-th dimension size as an integer.
- ```tf.shape(t)``` returns t's run-time shape as a tensor.
- An example:
```python
x = tf.placeholder(tf.float32, shape=[None, 8]) # x shape is non-deterministic while building the graph.
print(x.shape) # Outputs static shape (?, 8).
shape_t = tf.shape(x)
with tf.Session() as sess:
    print(sess.run(shape_t, feed_dict={x: np.random.random(size=[4, 8])})) # Outputs run-time shape (4, 8).
```
- [] (empty square brackets) as a shape denotes a scalar (0 dim). E.g. tf.FixedLenFeature([], ..) is a scalar feature.

#### Tensor contraction (more generalized matrix multiplication):
```python
# Matrix multiplication
tf.einsum('ij,jk->ik', m0, m1)  # output[i, k] = sum_j m0[i, j] * m1[j, k]

# Dot product
tf.einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

# Outer product
tf.einsum('i,j->ij', u, v)  # output[i, j] = u[i]*v[j]

# Transpose
tf.einsum('ij->ji', m)  # output[j, i] = m[i,j]

# Batch matrix multiplication
tf.einsum('aij,jk->aik', s, t)  # out[a, i, k] = sum_j s[a, i, j] * t[j, k]

# Batch tensor contraction
tf.einsum('nhwc,nwcd->nhd', s, t)  # out[n, h, d] = sum_w_c s[n, h, w, c] * t[n, w, c, d]
```

#### A typical input_fn (used for train/eval) for tf.estimator API:
```python
def make_input_fn(mode, ...):
    """Return input_fn for train/eval in tf.estimator API.

    Args:
        mode: Must be tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.EVAL.
        ...
    Returns:
        The input_fn.
    """
    def _input_fn():
        """The input function.

        Returns:
            features: A dict of {'feature_name': feature_tensor}.
            labels: A tensor of labels.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            features = ...
            labels = ...
        elif mode == tf.estimator.ModeKeys.EVAL:
            features = ...
            labels = ...
        else:
            raise ValueError(mode)
        return features, labels

    return _input_fn
```

#### A typical model_fn for tf.estimator API:
```python
def make_model_fn(...):
    """Return model_fn to build a tf.estimator.Estimator.

    Args:
        ...
    Returns:
        The model_fn.
    """
    def _model_fn(features, labels, mode):
        """Model function.

        Args:
            features: The first item returned from the input_fn for train/eval, a dict of {'feature_name': feature_tensor}. If mode is ModeKeys.PREDICT, same as in serving_input_receiver_fn.
            labels: The second item returned from the input_fn, a single Tensor or dict. If mode is ModeKeys.PREDICT, labels=None will be passed.
            mode: Optional. Specifies if this training, evaluation or prediction. See ModeKeys.
        """
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Calculate the predictions.
            predictions = ...
            # For inference/prediction outputs.
            export_outputs = {
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME: tf.estimator.export.PredictOutput({
                    'output_1': predict_output_1,
                    'output_2': predict_output_2,
                    ...
                }),
            }
            ...
        else:
            predictions = None
            export_outputs = None

        if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
            loss = ...
        else:
            loss = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = ...
            # Can use tf.group(..) to group multiple train_op as a single train_op.
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            export_outputs=export_outputs)

    return _model_fn
```

#### Use tf.estimator.Estimator to export a saved_model:
```python
# serving_features must match features in model_fn when mode == tf.estimator.ModeKeys.PREDICT.
serving_features = {'serving_input_1': tf.placeholder(...), 'serving_input_2': tf.placeholder(...), ...}
estimator.export_savedmodel(export_dir,
                            tf.estimator.export.build_raw_serving_input_receiver_fn(serving_features))
```

#### Use tf.contrib.learn.Experiment to export a saved_model:
```python
# serving_features must match features in model_fn when mode == tf.estimator.ModeKeys.PREDICT.
serving_features = {'serving_input_1': tf.placeholder(...), 'serving_input_2': tf.placeholder(...), ...}
export_strategy = tf.contrib.learn.utils.make_export_strategy(tf.estimator.export.build_raw_serving_input_receiver_fn(serving_features))
expriment = tf.contrib.learn.Experiment(..., export_strategies=[export_strategy], ...)
```

#### Load a saved_model and run inference (in Python):
```python
with tf.Session(...) as sess:
    # Load saved_model MetaGraphDef from export_dir.
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    # Get SignatureDef for serving (here PREDICT_METHOD_NAME is used as export_outputs key in model_fn).
    sigs = meta_graph_def.signature_def[tf.saved_model.signature_constants.PREDICT_METHOD_NAME]

    # Get the graph for retrieving input/output tensors.
    g = tf.get_default_graph()

    # Retrieve serving input tensors, keys must match keys defined in serving_features (when building input receiver fn).
    input_1 = g.get_tensor_by_name(sigs.inputs['input_1'].name)
    input_2 = g.get_tensor_by_name(sigs.inputs['input_2'].name)
    ...

    # Retrieve serving output tensors, keys must match keys defined in ExportOutput (e.g. PredictOutput) in export_outputs.
    output_1 = g.get_tensor_by_name(sigs.outputs['output_1'].name)
    output_2 = g.get_tensor_by_name(sigs.outputs['output_2'].name)
    ...

    # Run inferences.
    outputs_values = sess.run([output_1, output_2, ...], feed_dict={input_1: ..., input_2: ..., ...})
```

#### Build a tf.train.Example in Python:
```python
# ==================== Build in one line ====================
example = tf.train.Example(features=tf.train.Features(feature={
    'bytes_values': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[bytes_feature])),
    'float_values': tf.train.Feature(
        float_list=tf.train.FloatList(value=[float_feature])),
    'int64_values': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int64_feature])),
    ...
}))
# ==================== OR progressivly ====================
example = tf.train.Example()
example.features.feature['bytes_feature'].bytes_list.value.extend(bytes_values)
example.features.feature['float_feature'].float_list.value.extend(float_values)
example.features.feature['int64_feature'].int64_list.value.extend(int64_values)
...
```

#### Build a tf.train.SequenceExample in Python:
```python
sequence_example = tf.train.SequenceExample()

# Populate context data.
sequence_example.context.feature[
    'context_bytes_values_1'].bytes_list.value.extend(bytes_values)
sequence_example.context.feature[
    'context_float_values_1'].float_list.value.extend(float_values)
sequence_example.context.feature[
    'context_int64_values_1'].int64_list.value.extend(int64_values)
...

# Populate sequence data.
feature_list_1 = sequence_example.feature_lists.feature_list['feature_list_1']
# Add tf.train.Feature to feature_list_1.
feature_1 = feature_list_1.feature.add()
# Populate feature_1, e.g. feature_1.float_list.value.extend(float_values)
# Add tf.train.Feature to feature_list_1, if any.
...
```

#### [Example](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/example/example.proto#L88) is roughly a map of {feature_name: value_list}.

#### [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/example/example.proto#L292) is roughly a map of {feature_name: list_of_value_lists}.

#### To parse a SequenceExample:
```python
tf.parse_single_sequence_example(serialized,
    context_features={
        'context_feature_1': tf.FixedLenFeature([], dtype=...),
        ...
    },
    sequence_features={
        # For 'sequence_features_1' shape, [] results with [?] and [k] results with [?, k], where:
        # ?: timesteps, i.e. number of tf.Train.Feature in 'sequence_features_1' list, can be variable.
        # k: number of elements in each tf.Train.Feature in 'sequence_features_1'.
        'sequence_features_1': tf.FixedLenSequenceFeature([], dtype=...),
        ...
    },)
```

#### Writes seqeuence/iterator of tfrecords into multiple sharded files, round-robin:
```python
class TFRecordsWriter:
    def __init__(self, file_path):
        """Constructs a TFRecordsWriter that supports writing to sharded files.

        Writes a sequence of Example or SequenceExample to sharded files.
        Typical usage:
        with TFRecordsWriter(<file_path>) as writer:
            # tfrecords 
            writer.write(tfrecords)

        :param file_path: Destination file path, with '@<num_shards>' at the
        end to produce sharded files.
        """
        shard_sym_idx = file_path.rfind('@')
        if shard_sym_idx != -1:
            self._num_shards = int(file_path[shard_sym_idx + 1:])
            if self._num_shards <= 0:
                raise ValueError('Number of shards must be a positive integer.')
            self._file_path = file_path[:shard_sym_idx]
        else:
            self._num_shards = 1
            self._file_path = file_path

    def __enter__(self):
        if self._num_shards > 1:
            shard_name_fmt = '{{}}-{{:0>{}}}-of-{}'.format(
                len(str(self._num_shards)),
                self._num_shards)
            self._writers = [
                tf.python_io.TFRecordWriter(
                    shard_name_fmt.format(self._file_path, i))
                for
                i in range(self._num_shards)]
        else:
            self._writers = [tf.python_io.TFRecordWriter(self._file_path)]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writers:
            for writer in self._writers:
                if writer:
                    writer.flush()
                    writer.close()

    def write(self, tfrecords):
        """Writes a sequence/iterator of Example or SequenceExample to file(s).

        :param tfrecords: A sequence/iterator of Example or SequenceExample.
        :return:
        """
        if self._writers:
            for i, tfrecord in enumerate(tfrecords):
                writer = self._writers[i % self._num_shards]
                if writer:
                    writer.write(tfrecord.SerializeToString())
```

#### Visualize Tensorflow graph in jupyter/ipython:
```python
import numpy as np
from IPython import display

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped {} bytes>".format(size)
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph
        -basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display.display(display.HTML(iframe))
```
Then call ```show_graph(tf.get_default_graph())``` to show in your Jupyter/IPython notebook.
