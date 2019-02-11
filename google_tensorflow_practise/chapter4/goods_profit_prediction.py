"""
自定义损失函数-----以商品货物采购量例进行预测
"""
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_label = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义前向传播网络结构
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数
"""
损失函数1，loss_more = 1,loss_less=10
"""
# loss_more = 1
# loss_less = 10
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_label),
#                                (y - y_label)*loss_more,
#                                (y_label - y)*loss_less))

"""
损失函数2，loss_more = 10,loss_less 1
"""
# loss_more = 10
# loss_less = 1
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_label),
#                                (y - y_label)*loss_more,
#                                (y_label - y)*loss_less))

"""
损失函数3，MSE损失函数
"""
loss_more = 1
loss_less = 10
loss = tf.reduce_mean(tf.square(y-y_label))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 设置训练样本
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置Y，这里添加一个随机噪声，不然设置不同的损失函数意义不大
# 因为当预测完全正确时，损失函数都是最低
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 开始训练神经网络
with tf.Session() as sess:
    ops = tf.global_variables_initializer()
    sess.run(ops)
    STEPS = 5000
    for step in range(STEPS):
        start = (batch_size * step) % dataset_size
        end = start + batch_size # 超过dataset_size 的时候，由于索引方式，会自动选择选择到最后一项
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_label: Y[start:end]})
        if (step % 1000 == 0):
            # 每训练50次就打印一下损失函数
            print('loss', sess.run(loss, feed_dict={x: X[start:end], y_label: Y[start:end]}))
            print(sess.run(w1), '\n')
    print ("Final w1 is: \n", sess.run(w1))
