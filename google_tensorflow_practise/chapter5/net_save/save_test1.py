"""
tensorflow 模型持久化Api测试
"""
import tensorflow as tf

# 声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name = "v2")
ret = v1+v2

init_op = tf.global_variables_initializer()

# 初始化saver
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存
    saver.save(sess,"./save_path/model.ckpt")