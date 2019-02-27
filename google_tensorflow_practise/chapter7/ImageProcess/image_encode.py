"""
图像编码和解码函数

虽然说一张RGB图像是由一个三维数组组成，但是在图像存储时会将这个三维数组进行压缩编码处理，所以在读取图片的时候
需要对其进行解码处理
"""
import matplotlib.pyplot as plt
import tensorflow as tf


# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("./picture/pic.jpg",'rb').read()

with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。TensorFlow还提供了
    # tf.image.decode_png函数对png格式的图像进行解码。解码之后的结果为一个
    # 张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 输出解码之后的三维矩阵
    print(img_data.eval())

    # 使用pyplot工具可视化得到的图像。

    plt.imshow(img_data.eval())
    plt.show(block=False)

    # 将数据的类型转化为实数方便下面的样例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.uint8)

    # 将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中。
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./picture/pic_bak.jpg",'wb') as f:
        f.write(encoded_image.eval())
