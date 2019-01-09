"""
自定义损失函数
以商品利益预测为例
"""
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, (None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, (None, 1), name='y-input')

# 定义了一个单层的前向传播过程
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
# 自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), loss_more * (y - y_), loss_less * (y_ - y)))
# 使用MSE
# loss = tf.reduce_mean(tf.square(y-y_))
global_step = 5000
learning_rate = tf.train.exponential_decay(0.1,
                                           global_step,
                                           global_step/batch_size,0.9,staircase=True)
print()
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 生成训练集合
rdm = RandomState()
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 设置Y，这里添加一个随机噪声，不然设置不同的损失函数意义不大，因为当预测完全正确时，损失函数都是最低
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in X]

# 训练神经网络
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for steps in range(5000):
    start = steps * batch_size % dataset_size
    end = min(start + batch_size, dataset_size)
    sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    if steps % 100 == 0:
        print('After %d steps ,the weight is ' % steps, sess.run(w1))
        print('the loss is %f' % sess.run(loss,feed_dict={x:X[start:end],y_:Y[start:end]}))

sess.close()
