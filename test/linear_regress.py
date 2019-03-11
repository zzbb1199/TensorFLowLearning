"""
利用神经网络做线性回归拟合
y = 10*x+100
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络参数
SAMPLE_NUMS = 3000
INPUT_NODES = 1
HIDDEN_NODES = 500
OUTPUT_NODES = 1
TRAIN_STEPS = 10000

X = np.linspace(-1, 1, SAMPLE_NUMS)[:,np.newaxis]
noise = np.random.normal(0, 0.05, X.shape)
Y = X ** 2 + noise


# 定义输入层
with tf.name_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x-input')
    y_label = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')

# 定义隐含层
with tf.name_scope('fc-layer1'):
    weights = tf.Variable(tf.truncated_normal(shape=(INPUT_NODES, HIDDEN_NODES), stddev=0.01), name='weights')
    biases = tf.Variable(tf.constant([0], dtype=tf.float32, shape=[HIDDEN_NODES]), name='biases')
    layer1_output = tf.nn.relu(tf.matmul(x, weights) + biases)

with tf.name_scope('fc-layer2'):
    weights = tf.Variable(tf.truncated_normal(shape=(HIDDEN_NODES, OUTPUT_NODES), stddev=0.01), name='weights')
    biases = tf.Variable(tf.constant([0], dtype=tf.float32, shape=[OUTPUT_NODES], name='biases'))
    y_logit = tf.matmul(layer1_output, weights) + biases

# 定义损失函数，这里使用MSE
loss = tf.reduce_mean(tf.square(y_logit - y_label))
# 定义训练目标
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# everything has been done, so let's train this model
# create a figure to plot the progress of the regression
fig = plt.figure(1)
# 绘制原始散点图
plt.scatter(X,Y)
ax = fig.add_subplot(1,1,1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAIN_STEPS):
        # 取得一个batch的训练数据
        sess.run(train_step, feed_dict={x:X,y_label:Y})

        # every 300 steps, print the result of the loss
        if i % 300 == 0:
            print('after %d training steps, the loss value is %f' % (i,sess.run(loss,feed_dict={x:X,y_label:Y})))
            # plot the result
            line = plt.plot(X,sess.run(y_logit,feed_dict={x:X,y_label:Y}),'k-',lw=3)
            plt.pause(0.1)
            ax.lines.remove(line[0])

plt.show()
