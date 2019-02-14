"""
TensorFlow 实现CNN 测试
"""
import tensorflow as tf


input = tf.Variable(tf.truncated_normal((1,32,32,3),dtype=tf.float32))

# CNN的参数个数智能过滤器的尺寸、深度以及当前层节点矩阵的深度有关。
# 所有这声明的参数变量是一个四维矩阵，前面两个维度代表过滤器的尺寸，第三个维度
# 表示当前层的深度，第四个代表过滤器的深度
filter_weight = tf.get_variable(
    'weight',[5,5,3,16],
    initializer=tf.truncated_normal_initializer(stddev=0.1)
)

# CNN中biases也是共享的（和weights一样），总共有过滤器深度个偏置项
biases = tf.get_variable(
    'biases',[16],initializer=tf.constant_initializer(0.1)
)

# CNN卷积层的前向传播
conv = tf.nn.conv2d(
    input,filter_weight,strides=[1,1,1,1],padding='SAME'
)

# 添加偏置项
bias = tf.nn.bias_add(conv,biases)

# 将计算结果通过ReLu激活函数
actived_conv = tf.nn.relu(bias)

# 最大池化层
pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],
                      strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conved,biased,pool = sess.run([conv,bias,pool])
    print('conved size',conved.shape)
    print('biased size',biased.shape)
    print('pool size',pool.shape)