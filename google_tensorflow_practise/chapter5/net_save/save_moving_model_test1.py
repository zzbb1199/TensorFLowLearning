"""
Tensorflow提供的持久化Api中有恢复时重命名的功能
这样的功能对使用滑动平均的模型有很大便利性，因为tensorflow中
实现滑动平均使用了影子变量，通过恢复重命名的机制可以方便的将
影子变量映射到当前模型中来

本脚本给出一个简单的样例
"""
import tensorflow as tf

# step = tf.Variable(0,dtype=tf.float32)
v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有声明滑动平均模型的时候只有一个变量v
# 输出v:0
for var in tf.global_variables():
    print(var.name)

# 加上ema模型
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

# 声明ema模型后，再打印
# 输出v:0
# v/ExponentialMovingAverage:0
for var in tf.global_variables():
    print(var.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v, ema.average(v)]))

    # 保存时，会将v:0和v/ExponentialMovingAverage:0都保存
    saver.save(sess, "./save_path/model.ckpt")