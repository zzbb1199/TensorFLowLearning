"""
Tensorflow test programing
"""
import tensorflow as tf

a = tf.constant([1.0, 2.0], dtype=tf.float32, name="a")
b = tf.constant([1.0, 2.0], dtype=tf.float32, name="b")
ret = a+b

sess = tf.Session()
print(a)
print(b)
print(ret)
print(sess.run(ret))
sess.close()
