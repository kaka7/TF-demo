#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from API import convOpt
from API.templete import  Visualize_train,Save_and_load_mode
import os
FLAGS=None

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    # Merge all the summaries and write them out to /tmp/tensorflow/mnistDataSet/logs/mnist_with_summaries (by default)
    merged = Visualize_train.merge_all_summary()

    if not os.path.exists(os.path.join(FLAGS.log_dir + '/train')): os.makedirs(
        os.path.join(FLAGS.log_dir + '/train'))
    if not os.path.exists(os.path.join(FLAGS.log_dir + '/test')): os.makedirs(os.path.join(FLAGS.log_dir + '/test'))
    train_writer = Visualize_train.FileWriter_summary(os.path.join(FLAGS.log_dir + '/train'), sess.graph)
    test_writer = Visualize_train.FileWriter_summary(os.path.join(FLAGS.log_dir + '/test'))

    save = Save_and_load_mode(FLAGS.log_dir, sess)
    if not save.load_model(): init.run()
    for step in range(FLAGS.num_steps):
        batch_x, batch_y = input_model.inputs()
        train_op.run({x: batch_x, y_: batch_y})

        if step % FLAGS.disp_step == 0:
            acc = accuracy.eval({x: batch_x, y_: batch_y})
            print("step", step, 'acc', acc,
                  'loss', cross_entropy.eval({x: batch_x, y_: batch_y}))
            train_result = merged.eval({x: batch_x, y_: batch_y})
            train_writer.add_summary(train_result, step)

            test_x, test_y = input_model.test_inputs()
            acc = accuracy.eval({x: test_x, y_: test_y})
            print("step", step, 'acc', acc)
            test_result = merged.eval({x: test_x, y_: test_y})
            test_writer.add_summary(test_result, step)

        save.save_model(step)
    sess.close()

FLAGS=None
# def tf_model():
#     visual_model = Visualize_train()
#     with tf.name_scope('input'):
#         x = tf.placeholder(tf.float32, [None, 28 * 28 * 1], 'x')
#         y_ = tf.placeholder(tf.float32, [None, 10], 'y_')
#     with tf.name_scope('input_reshape'):
#         image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
#         visual_model.image_summary('input', image_shaped_input, 10)
#     # input_model = Inputs(FLAGS.data_dir, FLAGS.batch_size, one_hot=FLAGS.one_hot)/
#     imput = LoadDatas(FLAGS.data_dir, FLAGS.batch_size,).mnistDataSet
#
#     # weight init
#     with tf.name_scope('line_layer'):
#         with tf.name_scope('weights'):
#             w = tf.Variable(tf.random_normal([28 * 28 * 1, 10]))  # 二分类
#             visual_model.variable_summaries(w)
#         with tf.name_scope('biases'):
#             b = tf.Variable(tf.random_normal([10]))
#             visual_model.variable_summaries(b)
#
#     model = Model(x, y_, w, b, FLAGS.learning_rate)
#     with tf.name_scope('Wx_plus_b'):
#         y = model.forward_pred(activation='softmax')
#         visual_model.hist_summary('pred', y)
#
#     with tf.name_scope('total_loss'):
#         cross_entropy = model.loss(y, MSE_error=False, one_hot=FLAGS.one_hot)
#         visual_model.scalar_summary('cross_entropy', cross_entropy)
#
#     train_op = model.train(cross_entropy)
#
#     with tf.name_scope('accuracy'):
#         accuracy = model.evaluate(y, one_hot=FLAGS.one_hot)
#         visual_model.scalar_summary('accuracy', accuracy)
#
#     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     with tf.Session() as sess:
#         # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
#         merged = visual_model.merge_all_summary()
#
#         if not os.path.exists(os.path.join(FLAGS.log_dir + '/train')): os.makedirs(
#             os.path.join(FLAGS.log_dir + '/train'))
#         if not os.path.exists(os.path.join(FLAGS.log_dir + '/test')): os.makedirs(os.path.join(FLAGS.log_dir + '/test'))
#         train_writer = visual_model.FileWriter_summary(os.path.join(FLAGS.log_dir + '/train'), sess.graph)
#         test_writer = visual_model.FileWriter_summary(os.path.join(FLAGS.log_dir + '/test'))
#
#         save = Save_and_load_mode(FLAGS.log_dir, sess)
#         if not save.load_model(): init.run()
#         for step in range(FLAGS.num_steps):
#             batch_xs, batch_ys = input_model.inputs()
#             train_op.run({x: batch_xs, y_: batch_ys})
#
#             if step % FLAGS.disp_step == 0:
#                 acc = accuracy.eval({x: batch_xs, y_: batch_ys})
#                 print("step", step, 'acc', acc,
#                       'loss', cross_entropy.eval({x: batch_xs, y_: batch_ys}))
#                 train_result = merged.eval({x: batch_xs, y_: batch_ys})
#                 train_writer.add_summary(train_result, step)
#
#                 test_x, test_y = input_model.test_inputs()
#                 acc = accuracy.eval({x: test_x, y_: test_y})
#                 print("step", step, 'acc', acc)
#                 test_result = merged.eval({x: test_x, y_: test_y})
#                 test_writer.add_summary(test_result, step)
#
#             save.save_model(step)
#         sess.close()
#
#         """
#         # test acc
#         test_x,test_y=input_model.test_inputs()
#         print('test acc', acc)
#         """