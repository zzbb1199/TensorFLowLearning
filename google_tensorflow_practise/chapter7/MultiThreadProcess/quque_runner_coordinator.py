"""
使用QueueRunner和Coordinator实现多线程操作队列
"""
import tensorflow as tf

# 声明一个先进先出的队列
queue = tf.FIFOQueue(100,"float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用QueueRunner来创建多个线程运行队列的入队操作,[enqueue_op] *5 表示启动5个线程。每个操都是enqueue_op
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

# 将qr节点加入到计算图上指定的集合
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 声明协同类
    coord = tf.train.Coordinator()
    # 启动线程队列
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 获取队列的值
    for i in range(10):
        # 之所以[0] 是要看前期的入队操作，每次入队多少
        print(i,sess.run(out_tensor)[0])

    # 使用tr.train.Coordinator来停止所有线程
    coord.request_stop()
    coord.join(threads)