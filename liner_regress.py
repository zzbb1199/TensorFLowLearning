"""
tensorflow 入门，线性回归
"""
import tensorflow as tf
import numpy as np

# create some data
x_data = np.random.uniform(-10, 10, 1000)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
Weight = tf.Variable(tf.random_uniform([1], -10, 10))
bias = tf.Variable(tf.random_uniform([1], -2, 2))

y = Weight * x_data + bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tensorflow structure end

sess = tf.Session()
sess.run(init)
for step in range(1000):
    if step % 100 == 0:
        print('step:', step, 'loss:', sess.run(loss),
              'weight:', sess.run(Weight), 'bias', sess.run(bias))
    sess.run(train)
