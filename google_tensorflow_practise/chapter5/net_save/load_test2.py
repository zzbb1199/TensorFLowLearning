"""
将已经保存的结构加载出来，不需要重新加载网络结构
但是会把所有的参数加载进来
"""
import tensorflow as tf

# 直接加载持久化的图
saver = tf.train.import_meta_graph("./save_path/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess,"./save_path/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))