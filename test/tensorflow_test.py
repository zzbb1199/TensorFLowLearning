"""
Tensorflow test programing
"""
import tensorflow as tf


a = tf.Variable(tf.constant([0,0,1],shape=[3,1],dtype=tf.float32))
b = tf.Variable(a)
c = tf.Variable(tf.constant([0,1,0],shape=[3,1],dtype=tf.float32))

compare_a_b = tf.equal(tf.argmax(a,0),tf.argmax(b,0))
compare_a_c = tf.equal(tf.argmax(a,0),tf.argmax(c,0))

convert2float_a_b = tf.cast(compare_a_b,dtype=tf.float32)
convert2float_a_c = tf.cast(compare_a_c,dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("a = ",sess.run(a))
    print("b = ", sess.run(b))
    print("c = ", sess.run(c))
    print('argmax a',sess.run(tf.argmax(a,  axis=0)))
    print('argmax b', sess.run(tf.argmax(b, axis=0)))
    print('argmax c', sess.run(tf.argmax(c, axis=0)))
    print('compare a b',sess.run(compare_a_b))
    print('compare a c', sess.run(compare_a_c))
    print('convert a b',sess.run(convert2float_a_b))
    print('convert a c',sess.run(convert2float_a_c))
