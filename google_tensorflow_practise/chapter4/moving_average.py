"""
滑动平均模型测试
"""
import tensorflow as tf


v1 = tf.Variable(0,dtype=tf.float32)
# 定义step，用于控制滑动平均模型衰减的动态更新
step = tf.Variable(0,dtype=tf.float32)

# 定义一个滑动平均的类
ema = tf.train.ExponentialMovingAverage(0.99,step)
# 定义滑动操作
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average([v1])来获取滑动平均之后的变量取值
    print(sess.run([v1,ema.average(v1)]))

    # 更新v1的值到5
    sess.run(tf.assign(v1,5))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) # [5.0, 4.5]

    # 更新step，验证动态衰减
    sess.run(tf.assign(step,10000))
    # 更新v1的值位10
    sess.run(tf.assign(v1,10))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    # 再次更新滑动平均
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))