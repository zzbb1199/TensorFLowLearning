"""
利用tensorboard监控程序中的各项指标
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 将tensorboard日志输出到哪个目录下
SUMMARY_DIR = './log'
# BATCH
BATCH_SIZE = 100
# 训练迭代次数
TRAIN_STEPS = 30000

def variables_summaries(var,name):
    """ 生成变量监控信息并定义生成监控信息日志的操作。
    :param var:  需要记录的张量
    :param name: 在可视化结果中显示的图表名称。一般与变量名一直
    :return:
    """
    with tf.name_scope("summaries"):
        # 记录张量中元素的取值分布
        tf.summary.histogram(name,var)

        # 计算变量的平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name,mean)

        # 计算变量的标准差，并定义生成其日志的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)


def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
    """
    生成一层fcw网络
    :param input_tensor:
    :param input_dim:
    :param output_dim:
    :param layer_name:
    :param act:
    :return:
    """
    # 将同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 声明神经网络边上的权重，并生成对应的日志函数
            weights = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
            variables_summaries(weights,layer_name+'/weights')

        with tf.name_scope('biases'):
            # 生成神经网络的biases
            biases = tf.Variable(tf.constant(0.0,shape=[output_dim]))
            variables_summaries(biases,layer_name+'/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor,weights)+biases
            # 记录神经网络输出节点在经过激活函数之前的分布
            tf.summary.histogram(layer_name+'/preactivate',preactivate)

        activations = act(preactivate,name='activation')

        # 记录神经网络输出节点在经过激活函数之后的分布
        tf.summary.histogram(layer_name+'/activations',activations)
        return activations

def main(argv=None):
    mnist = input_data.read_data_sets('../datas/MNIST_data',one_hot=True)
    # 定义输入
    with tf.name_scope('input'):
        x = tf.placeholder(dtype=tf.float32,shape=(None,784),name='x-input')
        y_label = tf.placeholder(dtype=tf.float32,shape=(None,10),name='y-input')

    # 将输入向量还原成图片的像素矩阵,并将当前的图片信息日志输出
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input-image',image_shaped_input)

    hidden1 = nn_layer(x,784,500,'layer1')
    y = nn_layer(hidden1,500,10,'layer2')

    # 计算交叉熵并定义生成交叉熵监控日志的操作
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_label)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy',cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)

    # 计算准确率，并输出到日志
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_label,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
        # 输出accur
        tf.summary.scalar('accuracy',accuracy)

    # 定义执行所有日志操作
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log')
        writer.add_graph(tf.get_default_graph())
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行日志输出操作和训练操作
            summary,_ = sess.run([merged,train_step],
                                 feed_dict={x:xs,y_label:ys})
            # 写入得到的summary
            writer.add_summary(summary,i)

        writer.close()


if __name__ == '__main__':
    tf.app.run()
