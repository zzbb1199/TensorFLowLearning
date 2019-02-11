"""
Tensorflow的保存及提取
"""
import tensorflow as tf


def save_write():
    """
    save variable
    :return:
    """
    # Save to file
    W = tf.Variable([[1,2,3],[5,6,7]],dtype=tf.float32)
    b = tf.Variable([[1,2,3]],dtype=tf.float32)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 保存
        save_path = saver.save(sess,"./save_test/save_net.ckpt")
        print("Save to path",save_path)

def save_read():
    """
    restore
    :return:
    """
    W = tf.Variable(tf.zeros((2,3),dtype=tf.float32))
    b = tf.Variable(tf.zeros((1,3),dtype=tf.float32))

    # not need to initialize
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./save_test/save_net.ckpt")
        print("W",sess.run(W))
        print("b",sess.run(b))

if __name__ == '__main__':
    save_read()