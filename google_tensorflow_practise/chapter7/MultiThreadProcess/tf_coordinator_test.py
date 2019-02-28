"""
Tensorflow Coordinator类使用例程
"""
import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord,worker_id):
    """
    县城中运行的程序，这个程序每隔1miao判断是否需要停止并打印自己的ID
    :param coord:
    :param worker_id:
    :return:
    """
    # Coordinator类提供了判断当前线程是否应该停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.05:
            print("Stop from id %d" % worker_id)
            # 请求关闭所有线程
            coord.request_stop()
        else:
            print("working on id %d " % worker_id)
        time.sleep(1)

if __name__ == '__main__':
    # 声明一个线程协调类
    coord = tf.train.Coordinator()
    # 声明5个线程
    threads = [
        threading.Thread(target=MyLoop,args=(coord,i)) for i in range(5)
    ]
    # 启动所有线程
    for t in threads:
        t.start()
    # 等待所有线程退出
    coord.join(threads)