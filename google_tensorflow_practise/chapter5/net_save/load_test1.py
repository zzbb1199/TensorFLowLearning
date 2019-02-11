"""
将已经保存的模型重新加载出来,需要重新定义网络结构图
"""
import tensorflow as tf

# 声明网络结构
v1 = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[1]),name="v2")
ret = v1+v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"./save_path/model.ckpt")
    print(sess.run(ret))