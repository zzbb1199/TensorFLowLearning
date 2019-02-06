"""
过拟合解决方式：
1. 提供更多的训练数据
2. 采用L1，L2正则
3. Dropout 随机丢失神经元节点
"""

import tensorflow as tf

weight = tf.constant([1.0, -2.0],
                     [-3.0, 4.0])

with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weight)))
