import tensorflow as tf

# 随机给予初直
from numpy import int32, float32

# weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
# all_zeros = tf.Variable(tf.zeros([2, 3], dtype=int32))
# all_ones = tf.Variable(tf.ones([2, 3], dtype=int32))
# all_fill = tf.Variable(tf.fill([2, 3], 4))
# all_constant = tf.Variable(tf.constant([2, 3]))

# 使用其他变量来初始化新的变量
# w2 = tf.Variable(weights.initial_value())
# w3 = tf.Variable(weights.initial_value()*2)


# 下面为前向传播神经网络的代码
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))

# 输入向量
"""
注意这里定义1x2的矩阵的时候，用的是2维矩阵。具体是因为shape属性。
shape=(1,1)，也是2维数组。shape=(1,)才是一维数组，参与矩阵运算的时候需要使用2D
"""
x = tf.constant([[0.7, 0.9],[1.0,3.0]]) # 使用常量
# x = tf.placeholder(dtype=float32, name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 需要先初始化两个全职张量
# sess.run(w1.initializer)
# 全局变量初始化
init_op = tf.initialize_all_variables()
sess.run(init_op)
# 输出
print(sess.run(y))
# print(sess.run(y, feed_dict={x: [[0.7, 0.9],[0.7, 0.9],[0.7, 0.9]]}))
sess.close()
