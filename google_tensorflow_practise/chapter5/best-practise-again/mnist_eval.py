"""
mnist 验证训练好的模型
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import time

EVAL_INTERVAL_SECS = 5


def eval(mnist):
    """ 每隔EVAL_INTERVAL_SECS 验证一次训练好的神经网络
    :param mnist: 数据
    :return:
    """
    # 建立网络结构
    x = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference.INPUT_NODE))
    y_labels = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference.OUTPUT_NODE))
    y = mnist_inference.inference(x, None)

    # 验证目标
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    # 准备验证数据
    validation_datas = {x: mnist.validation.images,
                        y_labels: mnist.validation.labels}

    # 恢复滑动平均模型参数
    ema = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    vars_restore = ema.variables_to_restore()

    # 定义saver
    saver = tf.train.Saver(vars_restore)

    sess = tf.Session()
    # 开始验证
    while True:

        # 读取最新训练好的模型
        ckpt = tf.train.get_checkpoint_state(
            mnist_train.SAVE_PATH
        )
        if ckpt:
            # 加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名validation_datas获取迭代次数
            global_step = ckpt.model_checkpoint_path.split("\.")[-1].split("-")[-1]
            acur = sess.run(accuracy, feed_dict=validation_datas)
            print("After %s step(s) trianing, the accuracy is %g"
                  % (global_step, acur))

        time.sleep(EVAL_INTERVAL_SECS)
    # 释放资源
    sess.close()

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data",one_hot=True)
    eval(mnist)


if __name__ == '__main__':
    tf.app.run()