"""
mnist 5.2.1章节
实现mnist手写体识别总程序
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集合相关的常熟
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集来说，也就是一张图片的像素点数目
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。对应0-9 这10个数字

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数。
BATCH_SIZE = 100  # 一次使用多少个训练数据
LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.001  # 正则化系数
TRAINING_STEPS = 30000  # 训练次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weigths1, biases1, weigths2, biases2):
    """
    定义网络结构，返回前向传播的结果

    网络结构采用全连接的3层结构
    :param input_tensor: 输入向量
    :param avg_class: 滑动平均类
    :param weigths1: 1-2层weights
    :param biases1:
    :param weigths2: 1-2层weights
    :param biases2:
    :return: 返回前向传播的结果
    """
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weigths1) + biases1)
        # 注意下面没有加ReLU激活函数处理，因为在计算损失函数的时候会一并计算softmax函数
        return tf.matmul(layer1, weigths2) + biases2
    else:
        # 首先使用avg_class做滑动平均处理
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weigths1)) +
            avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weigths2)) + \
               avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, (None, INPUT_NODE), name="x-input")
    y_label = tf.placeholder(tf.float32, (None, OUTPUT_NODE), name="y-input")

    # 生成隐藏层的参数
    weigths1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weigths2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播结果，这里暂时不给出滑动平均处理
    y = inference(x, None, weigths1, biases1, weigths2, biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量
    # trainable=False
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY
                                                          , global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（如global_step就不需要了）。
    # tf.trainable_variables就是所有没有使用trainable=False的变量
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均的时候的前向传播
    y_average = inference(x, variable_averages, weigths1, biases1, weigths2, biases2)

    # 使用交叉熵来作为损失函数（MNIST本质上就一个10分类的问题）
    cross_entropy = tf.nn \
        .sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化使用函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失,一般只计算神经网络边上权重的正则化损失，不计算偏置
    regularization = regularizer(weigths1) + regularizer(weigths2)

    # 计算总的损失函数 = cross_entropy + regularization
    loss = cross_entropy_mean + regularization

    # 设置衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率
        global_step,  # 当前的迭代论数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率衰减速率
    )

    # 优化方向
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )

    # 每过一次数据都需要反向传播来更新网络中的参数，但是也需要更新滑动平均值。
    train_op = tf.group(train_step, variable_averages_op)

    # 不适用滑动平均
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_label,1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # 检验使用滑动平均模型的神经网络是否正确。
    correct_prediction_average = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_label, 1))
    # cast运行将bool数值变成实数
    accurary_average = tf.reduce_mean(tf.cast(correct_prediction_average, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的结果
        validate_feed = {x: mnist.validation.images,
                         y_label: mnist.validation.labels}

        # 准备测试数据。
        test_feed = {x: mnist.test.images,
                     y_label: mnist.test.labels}

        # 训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accurary, feed_dict=validate_feed)

                validate_acc_average = sess.run(accurary_average, feed_dict=validate_feed)
                print("After %d training step(s),validation accurary "
                      " is %g, using average accurary is %g"
                      % (i, validate_acc, validate_acc_average))

            # 产生这一轮将使用的batch数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs,
                                          y_label: ys})


        # 训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accurary,feed_dict=test_feed)
        print("After %d training step(s), test accurary using average"
              "model is %g" % (TRAINING_STEPS,test_acc))

# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类
    mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)
    train(mnist)

# Tensorflow 提供的一个入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
