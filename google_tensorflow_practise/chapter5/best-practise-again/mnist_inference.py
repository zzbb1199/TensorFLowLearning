"""
前向传播神经网络
"""
import tensorflow as tf

# 定义神经网络参数
INPUT_NODE = 28*28      # 28*28的像素点阵
LAYER1_NODE = 500       # 隐藏层神经元个数
OUTPUT_NODE = 10        # 输出层神经元，这里为10分类问题


def __get_weights(shape,regularizer=None):
    """ 获取层间权重值
    根据shape得到权重，并将正则化加入到"losses"集合中
    :param shape: 权重维度
    :param regularizer: 正则化函数
    :return: 层间权重
    """
    weights = tf.get_variable("weights",shape=shape,dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        # 添加正则化
        tf.add_to_collection("losses",regularizer(weights))
    return weights

def inference(input_tensor,regularizer=None):
    """ 定义神经网络结构，并得到前向传播的结果
    :param input_tensor: 输入张量
    :param regularizer: 正则化函数
    :return: 前向传播的结果
    """

    # 定义第一层结构
    with tf.variable_scope("hidden_layer"):
        weights = __get_weights([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[LAYER1_NODE]))
        # 得到隐藏层输出，这里使用了ReLU输出函数
        tmp_y = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    # 定义第二层结构
    with tf.variable_scope("output_layer"):
        weights = __get_weights([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[OUTPUT_NODE]))
        # 输出，因为后期要添加softmax层处理，所以这里不需要使用relu激活函数
        y = tf.matmul(tmp_y,weights)+biases

    return y