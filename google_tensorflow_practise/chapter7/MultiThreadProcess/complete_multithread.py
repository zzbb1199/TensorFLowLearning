"""
完整的多线程处理框架
"""

import tensorflow as tf
from complete_image_process import *

# 创建文件列表,通过正则表达式来匹配所有需要用来训练的数据（这些数据已经被持久化在TFRcord文件中）
files = tf.train.match_filenames_once("./tfreccords-*")
# 文件读取队列,num_epochs表示队列循环读取多少次
filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)     # 后续再统一将数据随机扰乱

# 解析文件
reader = tf.TFRecordReader()
_, serialized_examples = reader.read(filename_queue)
# 定义单个样例的解析操作
features = tf.parse_single_example(
    serialized_examples,
    features={
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64)
    }
)
# 取得需要的数据
image, label, height, width, channel = features['image'], features['label'], features['height'], features['width'], \
                                       features['channel']

# 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channel])
# 定义神经网络输入层图片的大小
image_size = 299
# 生成扰乱图片数据
distorted_image = preprocess_for_train(
    decoded_image,image_size,image_size,None
)

# 将处理后的图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue +3 * batch_size
image_batch,label_batch = tf.train.shuffle_batch(
    [distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue
)

# 定义神经网络的结构以及优化过程。image_batch可以作为输入提供给神经网络的输入层
# label_batch则提供给输入batch中样例的正确答案
logits = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    # 神经网络训练准备工作。这些工作包括变量初始化、线程启动
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # 神经网络训练过程
    for i in range(STEPS):
        sess.run(train_step)

    coord.request_stop()
    coord.join(threads)