"""
多线程队列读取TFRecord
"""
import tensorflow as tf


# 使用tf.train.match_filename_once函数获取文件列表
files = tf.train.match_filenames_once("./tfrecords-*")

# 通过tf.train.string_input_producer函数创建输入队列，输入队列中的
# 文件列表为tf.train.match_filename_once函数获取的文件列表，这里讲shuffle
# 参数设为False来避免随机打乱文件的顺序。但一般解决真实问题时，会讲shuffle参数设置为True
filename_queue = tf.train.string_input_producer(files,shuffle=False)

# 读取并解析一个样本
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i':tf.FixedLenFeature([],dtype=tf.int64),
        'j':tf.FixedLenFeature([],dtype=tf.int64)
    }
)
with tf.Session() as sess:
    # 虽然本程序中没有申明任何变量，但是match_filenames_once需要使用一些变量
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run([init_g,init_l])
    # 输出有哪些文件需要读取
    print(sess.run(files))

    # 生命tf.train.Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # 多次执行获取数据的操作
    for i in range(4):
        print(sess.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)