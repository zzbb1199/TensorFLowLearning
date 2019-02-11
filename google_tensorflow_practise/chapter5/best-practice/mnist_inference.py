# -*- codding:utf-8 -*-
"""
mnist 前向传播脚本
"""
import tensorflow as tf

# 定义神经网络的相关参数
INPUT_NODE = 28*28
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weigth_variable(shape,regularizer):
    weights = tf.get_variable(
        "weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    # 加入正则化损失
    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))

    return weights

# 定义前向传播的过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope("layer1"):
        weights = get_weigth_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=
                                 tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)


    # 声明第二层
    with tf.variable_scope("layer2"):
        weights = get_weigth_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=
                                 tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1,weights)+biases

    return layer2

