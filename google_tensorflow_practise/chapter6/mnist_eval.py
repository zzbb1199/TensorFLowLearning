# -*- coding:utf-8 -*-
"""
mnist 滑动平均模型测试
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference和mnist_train中定义的参数
import mnist_inference
import mnist_train
import numpy as np

# 没10s加载一次最新的模型，病在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:

        xs = mnist.validation.images
        reshaped_xs = np.reshape(xs, [xs.shape[0],
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.NUM_CHANNELS])

        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [xs.shape[0],
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS], name="x-input")
        y_label = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

        validate_feed = {x:reshaped_xs,
                         y_label: mnist.validation.labels}

        # 调用封装好的函数来计算前向传播结果
        y = mnist_inference.inference(x,False, None)

        # 使用前向传播的结果计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variabels_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variabels_to_restore)

        # 每隔EVAL_INTERVAL_SEC秒调用一次计算正确率的过程以检验训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                        .split('-')[-1]
                    acurracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, acurracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
