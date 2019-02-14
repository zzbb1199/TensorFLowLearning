# -*- coding: utf-8 -*-
"""
mnist 训练脚本
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 加载mnist_inference中定义的常量和前向传播的函数
import mnist_inference

# 配置神经网络参数

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model_save_path/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32,
                       shape=(BATCH_SIZE,
                              mnist_inference.IMAGE_SIZE,
                              mnist_inference.IMAGE_SIZE,
                              mnist_inference.NUM_CHANNELS),
                       name="x-input")
    y_label = tf.placeholder(tf.float32,
                             shape=(None,
                                    mnist_inference.NUM_LABELS),
                             name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，学习率，滑动平均操作以及训练过程
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 得到总损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(
        loss,
        global_step=global_step
    )
    # 更新滑动平均
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # vars = tf.trainable_variables()
    variable_average_op = variable_average.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 开始训练
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # reshap xs
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_label: ys})
            # 每1000轮就保存一次模型
            if i % 100 == 0:
                # 输出当前训练情况
                print("Afterh %d training step(s), loss on training batch "
                      "is %g" % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step 参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
                # 比如"model.ckpt-1000" 表示训练1000轮以后的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
