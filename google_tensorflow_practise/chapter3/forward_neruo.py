"""
神经网络前向传播算法
"""
import tensorflow as tf

# 定义w1,w2两个变量
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))


# 定义输入
x = tf.constant([0.7,0.9],shape=(1,2))
print(x.shape)
# 前相传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

##  运算 ##
sess = tf.InteractiveSession()
# 初始化所有变量！！！！！！！！！！！！！！！！！！！！！！很容易忘记
ops = tf.global_variables_initializer()
sess.run(ops)
print(sess.run(y))
sess.close()

