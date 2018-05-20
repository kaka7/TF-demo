#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
input_data.read_data_sets().train.next_batch()

class LoadDatas(object):
    def __init__(self,datas_store_path,batch_size_num,one_hot=True):
        self.datas_store_path=datas_store_path
        self.batch_size_num=batch_size_num
        self.mnistDataSet=input_data.read_data_sets(self.datas_store_path, one_hot=one_hot)
    def batch_train_datas(self):
        batch_x, batch_y = self.mnistDataSet.train.next_batch(self.batch_size_num)
        return batch_x, batch_y
    def all_test_datas(self):
        return self.mnistDataSet.test.images,self.mnistDataSet.test.labels
    def reshape_data(self,mnistDataSet):
        return tf.reshape(mnistDataSet, [-1, 28, 28, 1])

