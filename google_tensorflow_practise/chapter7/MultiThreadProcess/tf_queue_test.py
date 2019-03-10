"""
Tensorflow中的队列操作
"""
import tensorflow as tf

# 创建一个FIFO队列，指定队列中最多可以保存两个元素
queue = tf.FIFOQueue(2,dtypes="int32")
# 使用enqueue_many函数来初始化队列中元素。和变量初始话类似，在使用队列之前需要明确的调用这个初始话过程
init = queue.enqueue_many(([1,10],))
# Dequeue函数将队列中第一个元素出列队。这个元素的值被存在变量x中
x = queue.dequeue()
# 将得到的值加1
y = x+1
# 将加1后的值在重新加入队列
q_inc = queue.enqueue([y])

with tf.Session() as sess:
    # 运行初始话操作
    init.run()
    for i in range(6):
        v,_ = sess.run([x,q_inc])
        print(v)