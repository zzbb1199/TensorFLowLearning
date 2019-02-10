"""
mnist 神经网络训练脚本
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference  # 导入mnist中的函数
import matplotlib.pyplot as plt
import numpy as np

# 定义神经网络中的部分参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_DECAY = 0.99        # 学习率衰减率
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化比率
TRAINING_STEPS = 30000
BATCH_SIZE = 100
SAVE_PATH = "../model_save_path/"
SAVE_NAME = "model.ckpt"


def train(mnist):
    """ 训练神经网络
    :param mnist: mnist数据集合
    :return: 无
    """

    # 定义网络结构
    # 输入
    x = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference.INPUT_NODE), name="x-input")
    y_labels = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference.OUTPUT_NODE), name="y-input")

    # 前向传播结构
    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)

    # 定义损失函数
    # 由于是分类问题，这里采用交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_labels,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 根据在mnist_inference中加入的正则化损失集合，可以得到总损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 优化目标
    # 定义动态衰减学习率
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_DECAY,
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step)

    # 上述已经完成了原始的网络结构，下面加入滑动平均模型

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
    maintain_ops = ema.apply(tf.trainable_variables())

    # 使滑动平均生效
    train_op = tf.group(train_step,maintain_ops)


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver()

        # 绘图变量初始化
        loss_save = np.zeros(shape=(TRAINING_STEPS),dtype=np.float32)
        # 开始训练
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            training_datas = {
                x: xs,
                y_labels:ys
            }
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict=training_datas)
            loss_save[i] = loss_value
            if i % 1000 == 0:
                # 每训练500次，打印一下结果，并把模型保存一下
                print("After %d training step(s), loss on this model is %g"
                      % (step,loss_value))
                # 保存
                saver.save(sess,SAVE_PATH+SAVE_NAME,global_step=global_step)
    x = np.arange(1,TRAINING_STEPS+1,1)
    # 训练完成
    plt.plot(x,loss_save,'b-',lw=2)
    plt.show()

def main(argv=None):
    # 读取数据
    mnist = input_data.read_data_sets("../MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()