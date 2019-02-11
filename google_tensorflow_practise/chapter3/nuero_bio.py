"""
完整的神经网络代码
@author:Raven
@date:2019-1-4
"""
import tensorflow as tf
from numpy.random import RandomState

# 定义batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape
# shape 设置为None,x方便测试不同的batch大小,在测试的时候可以把整个训练集放入到一个batch中
# 但是实际训练应该把batch设置成较小的size
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络强向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数---这里使用交叉熵
# cross_entropy = -tf.reduce_mean(
#     y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
# )
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
loss = tf.reduce_mean(cross_entropy)
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 定义训练样本
# 这里使用随机数代替
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则给出样本的标签.在这里所有的x1+x2 < 1 被认为是正样本(合格样本),否则为负样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]


# 创建一个Session来运行TesnsorFlow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量
    sess.run(init_op)
    print("before training:")
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)  # 可能最后一次超出dataset_size

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))

    print("end of training:")
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
