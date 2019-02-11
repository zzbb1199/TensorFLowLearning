"""
只加载部分参数,通过提供默认列表
本脚本提供两个测试：
1. 通过key加载部分变量
2. 通过字典序列加载部分变量的同时，重命名变量
"""
import tensorflow as tf


def load_some_vars_test():
    # 注意这里Tensorflow在计算图中寻找的是name="xx"的关键字，而不是变量名
    v1 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32), name="v1")
    saver = tf.train.Saver([v1])
    with tf.Session() as sess:
        saver.restore(sess, "./save_path/model.ckpt")
        print(sess.run(v1))


def load_some_vars_and_rename():
    rename_v1 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32), name="rename-v1")
    rename_v2 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32), name="rename-v2")
    saver = tf.train.Saver({"v1": rename_v1, "v2": rename_v2})
    with tf.Session() as sess:
        saver.restore(sess,"./save_path/model.ckpt")
        print(sess.run(rename_v1))
        print(sess.run(rename_v1.name))

if __name__ == '__main__':
    load_some_vars_and_rename()