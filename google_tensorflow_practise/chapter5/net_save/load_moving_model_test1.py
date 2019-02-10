"""
读取滑动平均模型
这里使用了读取重命名
"""
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 方法一：手写映射
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
# 方法二：Tensorflow提供相关的api
ema = tf.train.ExponentialMovingAverage(0.99)
saver = tf.train.Saver(ema.variables_to_restore())


with tf.Session() as sess:
    saver.restore(sess,"./save_path/model.ckpt")
    print(sess.run(v))
