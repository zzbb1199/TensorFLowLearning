"""
组合batch数据，方便输入到神经网络中进行训练
主要会使用到的函数
tf.train.batch
tf.train.shuffle_batch
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

# 下面是本脚本的核心内容
# 假设i 是样本的特征向量，j是该特征向量对应的标签
example,label = features['i'],features['j']

# 一个batch中样例的个数
batch_size = 3
# 组合样例的队列中最多可以存储的样例个数。一般来说这个队列的容量和batch_size有关。如果设置过大，会占用很多内存，如果设置过小，又会
# 因为没有element而block
capacity = 1000 + 3*batch_size

# 使用tf.train.batch函数来组合样例
example_batch,label_batch = tf.train.shuffle_batch(
    [example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=30
)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # 多线程处理
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # 获取并打印组合之后的样例。在真实问题中，这个输出一般会作为神经网络的输入
    for i in range(2):
        cur_example_batch,cur_label_batch = sess.run(
            [example_batch,label_batch]
        )
        print(cur_example_batch,cur_label_batch)

    coord.request_stop()
    coord.join(threads)