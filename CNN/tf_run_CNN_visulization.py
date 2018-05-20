#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import numpy as np
import tensorflow as tf
import os, sys
# matmul

#os.chdir("" )
# print("当前工作目录 : %s" % os.getcwd())
#一，加载数据，定义超参数
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/naruto/PycharmProjects/data/mnist",one_hot=True)
trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
# 原始的是[6000,784]
trX=trX.reshape([-1,28,28,1])
teX=teX.reshape([-1,28,28,1])
batch_size=128
test_size=256
strides=2
display_step=10
iter_batch_num=100
FLAGE=None
output_path="/home/naruto/PycharmProjects/MyPro/CNN/output"

#二，构建模型
# 定义edge
X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 定义模型结构
# init net parameter
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X,W1,W2,W3,W4,W_O,strides,p_keep_conv,p_keep_hidden):
    X=tf.reshape(X, shape=[-1, 28, 28, 1])
    with tf.name_scope("layer1"):
        # layer1 6000 28 28 1 ;3 3 1 32 ->6000 28 28 32 这里的same使得最后要补两个0
        l1a=tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME'))#6000 28 28 32
        l1b=tf.nn.max_pool(l1a,ksize=[1,strides,strides,1],strides=[1,strides,strides,1],padding='SAME')#14 14 32
        l1=tf.nn.dropout(l1b,p_keep_conv)
        tf.summary.histogram('W',W1)
    # layer2
    with tf.name_scope("layer2"):

        l2a = tf.nn.relu(tf.nn.conv2d(l1,W2,strides=[1,1,1,1],padding='SAME'))#14 14 64
        l2b = tf.nn.max_pool(l2a,ksize=[1,strides,strides,1],strides=[1,strides,strides,1],padding='SAME')# 7 7 64
        l2=tf.nn.dropout(l2b,p_keep_conv)
        tf.summary.histogram('W2', W2)

    #layer3
    with tf.name_scope("layer3"):

        l3a=tf.nn.relu(tf.nn.conv2d(l2,W3,strides=[1,1,1,1],padding='SAME'))#7 7 128
        l3b=tf.nn.max_pool(l3a,ksize=[1,strides,strides,1],strides=[1,strides,strides,1],padding='SAME')# 4 4 128
        l3c=tf.reshape(l3b,[-1,W4.get_shape().as_list()[0]])
        l3=tf.nn.dropout(l3c,p_keep_conv)
        tf.summary.histogram('W3', W3)

    #layer4
    with tf.name_scope("layer4"):

        l4a=tf.nn.relu(tf.matmul(l3,W4))
        l4=tf.nn.dropout(l4a,p_keep_hidden)
        tf.summary.histogram('W4', W4)

    #output
    with tf.name_scope("output"):

        pyx=tf.matmul(l4,W_O)
        tf.summary.histogram('W_O',W_O)

    return pyx

W1=init_weights([3,3,1,32])
W2=init_weights([3,3,32,64])
W3=init_weights([3,3,64,128])
W4=init_weights([4*4*128,625])#flate后FC,所以输入是上一层神经元数
W_O=init_weights([625,10])

py_x=model(X,W1,W2,W3,W4,W_O,strides,p_keep_conv,p_keep_hidden)
with tf.name_scope("train_cost"):
# 定义损失
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
    tf.summary.scalar('cost', cost)

# 定义优化方法
# optimize=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
optimize=tf.train.AdamOptimizer().minimize(cost)
predict_op=tf.argmax(py_x,1)
with tf.name_scope("train_correct_rate"):
    correct_rate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(py_x,1),tf.argmax(Y,1)),tf.float32))
    tf.summary.scalar('correct_rate', correct_rate)

#三，训练模型
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()

with tf.Session() as sess:
    writer=tf.summary.FileWriter(output_path,sess.graph)
    merge=tf.summary.merge_all()
    tf.global_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() # get last global_step
    print("Start from:", start)
    for i in range(iter_batch_num):
        # @method1
        # testing_batch=zip(np.arange(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        # for start,end in testing_batch:
        #     # 训练时是训练优化器
        #     sess.run(optimize,feed_dict={X:trX[start:end],Y:trY[start:end],p_keep_conv:0.75,p_keep_hidden:0.75})
        # @method2
        batch=mnist.train.next_batch(batch_size)
        # sess.run(optimize,feed_dict={X:batch[0].reshape([-1,28,28,1]),Y:batch[1].reshape([-1,28,28,1]),p_keep_conv:0.75,p_keep_hidden:0.75})
        # testData=tf.reshape(batch[0],shape=[-1,28,28,1])
        _ , train_corr_rate,train_cost=sess.run([optimize,correct_rate,cost],feed_dict={X: batch[0], Y: batch[1], p_keep_conv: 0.75,p_keep_hidden: 0.75})

        global_step.assign(i).eval()  # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        if i%display_step==0:
            print ("iter {}: the correct_rate is {}".format(i, sess.run(correct_rate,feed_dict={X: batch[0], Y: batch[1],p_keep_conv:0.75,p_keep_hidden:0.75})))
            result=sess.run(merge,feed_dict={X: batch[0], Y: batch[1], p_keep_conv: 0.75,p_keep_hidden: 0.75})
            writer.add_summary(result,i)
    print "Optimization Completed"

#四，评估，预测
    test_batch_data = mnist.test.next_batch(test_size)
    print ("the test data sets' correct_rate is {}".format(sess.run(correct_rate, feed_dict={X: test_batch_data[0], Y: test_batch_data[1],
                                                                                         p_keep_conv: 0.75,
                                                                                         p_keep_hidden: 0.75})))
    # 四，评估，预测

        # test_indices=np.arange(len(teX))
        # np.random.shuffle(test_indices)
        # test_indices=test_indices[0:test_size]
        # # 测试是带入训练好的模型以及测试数据
        # print (i ,np.mean(np.argmax(teY[test_indices],axis=1)==sess.run(predict_op,feed_dict={X:teX[test_indices],p_keep_conv: 1.0,
        #                                                  p_keep_hidden: 1.0})))

