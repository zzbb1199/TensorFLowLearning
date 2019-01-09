"""
神经网络学习率优化，避免神经网络在GradientDescent中来回震荡
"""
import tensorflow as tf

global_step = tf.Variable(0)

# 通过exponential_decay函数生成学习率,可跳转函数去看计算公式
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
# 定义自己的loss
loss = tf.reduce_mean(tf.ones([2, 2]))
# 使用指数衰减的学习率。
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(global_step=global_step, loss=loss)
