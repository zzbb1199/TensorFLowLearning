"""
图像大小调整:
一般来说，网络上获取的图像大小是不固定的，但神经网络输入节点的个数是固定的。
"""

import tensorflow as tf
import matplotlib.pyplot as plt


with tf.Session() as sess:
    # 读取原始图片
    image_data_raw = tf.gfile.FastGFile("./picture/pic.jpg",'rb').read()
    # 解码
    image_data = tf.image.decode_jpeg(image_data_raw)
    # 转换数据格式
    image_data = tf.image.convert_image_dtype(image_data,dtype=tf.float32)
    # 输出原始图片
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(image_data.eval())

    # 重新调整大小
    resized_img = tf.image.resize_images(image_data,[300,300],method=0)
    # 输出调整后的图片
    plt.subplot(1,2,2)
    plt.imshow(resized_img.eval())
    plt.show()

    # 输出调整后图像的大小。
    print(resized_img.get_shape())



