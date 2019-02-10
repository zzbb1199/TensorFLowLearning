"""
Tensorflow test programing
"""
import tensorflow as tf

a = tf.constant([10.0, 2.0,3,312,4,5], dtype=tf.float32, name="a")
# b = tf.constant([1.0, 2.0], dtype=tf.float32, name="b")
# ret = a+b

sess = tf.Session()
print(sess.run(tf.argmax(a,0)))
