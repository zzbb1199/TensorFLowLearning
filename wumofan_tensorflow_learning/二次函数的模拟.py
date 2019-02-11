"""
二次函数模拟，plot演示
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 产生模拟数据
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = x_data ** 10 + 0.2 + noise

# 绘制原始数据的散点图
fig = plt.figure(1)

ax = fig.add_subplot(2,1,1)
ax.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('ML visual')


# 通过机器学习，学习其参数
INPUT_NODE = 1
LAYER1_NODE = 30
OUTPUT_NODE = 1
LEARNING_RATE = 0.1

# 定义网络结构
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, 1),name="x-input")
    y_label = tf.placeholder(tf.float32, shape=(None, 1),name="y-input")


with tf.name_scope("layer"):
    weights1 = tf.Variable(tf.random_normal(shape=[INPUT_NODE, LAYER1_NODE],
                                        stddev=0.1),name="w1")
    biases1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]),name="b1")
    weights2 = tf.Variable(tf.random_normal(shape=[LAYER1_NODE, OUTPUT_NODE],
                                        stddev=0.1),name="w2")
    biases2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]),name="b2")

tmp_y = tf.nn.relu(tf.matmul(x, weights1) + biases1)
y = tf.matmul(tmp_y, weights2) + biases2

# 定义损失函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - y_label))
# 定义优化目标
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs/",sess.graph)

    STEPS = 300
    for i in range(STEPS):
        sess.run(train_step, feed_dict={x: x_data,
                                        y_label: y_data})
        if i % 100 == 0:
            loss_value, prediction = sess.run([loss, y]
                                              , feed_dict={x: x_data, y_label: y_data})
            print("After %d step(s) training, loss is %f" %
                  (i, loss_value))
            # 可视化训练过程
            line = ax.plot(x_data, prediction,'r-',lw=2)
            plt.pause(0.01)
            ax.lines.remove(line[0])

plt.show()
