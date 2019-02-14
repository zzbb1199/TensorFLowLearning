# -*-coding:utf-8 -*-

import tensorflow as tf

# 配置神经网络参数
INIPUT_NODE = 28 * 28
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷接层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512


# 定义CNN的前向传播过程
def inference(input_tensor, train, regularizer):
    """
    :param input_tensor:
    :param train: 区分训练还是测试过程
    :param regularizer:
    :return:
    """
    # 卷积层1
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            'weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            'bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0)
        )

        # 使用边长为5，深度位32的过滤器，strides=1,且使用全0填充
        conv1 = tf.nn.conv2d(
            input_tensor,conv1_weights,[1,1,1,1],padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    # 池化层1
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )

    # 卷积层2
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            'bias',[CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )

        # 卷积
        conv2 = tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    # 池化2
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )

    # flatten 操作
    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出7*7*64的矩阵
    pool_shape = pool2.get_shape().as_list()
    # 计算将矩阵拉直后向量的长度,pool_shape[0]位batch的大小
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

    # 通过tf.reshape函数变形
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    # 全连接前向传播，注意这里使用了dropout来避免过拟合
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weights",[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc1_biases = tf.get_variable(
            "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1)
        )
        # 只有全连接层的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection("losses",regularizer(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        # 加入dropout 避免过拟合
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    # 声明第六层全连接层的变量并实现前向传播过程。这一层的输入长度为512的向量
    # 输出为一组长度为10的向量
    with tf.variable_scope("layer6-fc2"):
        fc2_weigths = tf.get_variable(
            "weight",[FC_SIZE,NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc2_biases = tf.get_variable(
            "bias",[NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        # 加入正则
        if regularizer:
            tf.add_to_collection("losses",regularizer(fc2_weigths))
        logits = tf.matmul(fc1,fc2_weigths)+fc2_biases

    # 返回
    return logits