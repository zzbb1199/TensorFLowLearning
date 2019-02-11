"""
过拟合解决方式：
1. 提供更多的训练数据
2. 采用L1，L2正则
3. Dropout 随机丢失神经元节点

程序给出了L1，L2正则化的使用方式
之后给出了在复杂网络结构中如何计算含有正则化的损失函数。
新损失函数的计算公式= 原损失函数+lamda*R(w)
L1:R(w) = sum|w|
L2:R(w) = sum|w^2|
"""

import tensorflow as tf


def regularizer_test():
    """
    正则化函数测试，没有什么实际作用
    :return:
    """
    weight = tf.constant([[1.0, -2.0],
                          [-3.0, 4.0]])
    with tf.Session() as sess:
        # 正则一
        print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weight)))
        # 正则二
        print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weight)))


############################################################
# 下面给出在复杂网络结构中如何计算损失函数的样例程序                #
############################################################

def get_weight(shape,lamda):
    """ 生成神经网络一层的权重变量，并将权重通过L2正则加入到losses集合中
    :param shape: 权重维度
    :param lamda: L2正则比例
    :return: 权重
    """
    # 生成一个权重变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    # 将上述权重变量加入到损害函数losses集合中，方便后续管理
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamda)(var))
    # 返回权重
    return var

"""
主程序
"""
x = tf.placeholder(dtype=tf.float32,shape=(None,2))
y_label = tf.placeholder(dtype=tf.float32,shape=(None,1))
batch_size = 8

# 定义每一层中的节点个数
layer_dimension = [2,10,10,10,1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 当前迭代到网络中的第几层的输出
# 最开始应该就等于x，即输入
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 生成网络结构
for i in range(1,n_layers):
    # 下一层的维度
    out_dimension = layer_dimension[i]
    # 两层之间的权重
    weight = get_weight([in_dimension,out_dimension],0.001)
    # 偏置项
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    # 得到输出，这里使用了激活函数(ReLU)
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    # 更新in_dimension 为下一迭代做准备
    in_dimension = out_dimension

"""
计算损失函数
"""
# 损失函数第一部分--不加正则化的损失函数
loss1 = tf.reduce_mean(tf.square(cur_layer - y_label))
tf.add_to_collection("losses",loss1)
# 损失函数第二部分--加入到集中的所有正则化项

# 将第一部分和第二部分相加，得到最终的损失函数
loss = tf.add_n(tf.get_collection("losses"))