"""
完整图片处理样例
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image,color_ordering=0):
    """ 扰乱颜色
    :param image: 图片
    :param color_ordering: 扰乱次序。因为调整亮度、对比度、饱和度和色相的顺序会影响到最后得到的结果 ，
                            所以可以定义多种不同的顺序。具体使用哪一种顺序可以在训练数据预处理随机的选择的一种。
                            这样可以进一步降低无关因素对模型的影响。
    :return:
    """
    if color_ordering < 0 or color_ordering > 4:
        raise Exception("color_ordering error: color order less than 0 or greater than 4")
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255)
    elif color_ordering == 4:
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    return tf.clip_by_value(image,0,1)

def preprocess_for_train(image,height,width,bbox=None):
    """ 给定一张解码后的图像、目标图像的尺寸以及图像的标注框，此函数可以对给出的图像进行预处理。这个函数的输入图像
    是图像识别问题中原始的训练图像，而输出则是神经网络模型的输入层。注意这里只是处理模型的训练数据，对于预测的数据，
    一般不需要使用随机变换的步骤

    图像处理流程： 根据标注框随机截取图像，之后放缩到原始图像的大小，然后进行亮度，色相，对比度，饱和度的随机调整。
    :param image: 解码后的图像
    :param height: 图像高度
    :param width: 图像的宽度
    :param bbox:  标注框
    :return:
    """
    # 如果没有给定bbox，则认为整个图像就是需要关注的部分
    if bbox == None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    # 随机截取图像，减少需要关注的物体大小对图像识别算法的 影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),bbox
    )
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    # 将随机截取的图像调整为神经网络层输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(
        distorted_image,[height,width],method=np.random.randint(4)              # 调整是否有问题？？？？
    )
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像的色彩
    distorted_image = distort_color(distorted_image,np.random.randint(4))
    # 返回处理完的数据
    return distorted_image

if __name__ == '__main__':
    with tf.Session() as sess:
        # 读取原始图片数据
        image_raw_data = tf.gfile.GFile("./picture/pic.jpg",'rb').read()
        # 解码
        image_data = tf.image.decode_jpeg(image_raw_data)
        origin_shape = image_data.eval().shape
        # 图像数据转换为实数类型，方便后续处理
        image_data = tf.image.convert_image_dtype(image_data,dtype=tf.float32)
        # 增加维度，便于后续加入标注框
        image_data_with_box = tf.expand_dims(image_data,0)
        # 初始话标注框
        bbox = tf.constant([0.05,0.05,0.8,0.7],dtype=tf.float32,shape=[1,1,4])
        # 显示加入标注框的图像
        image_data_with_box = tf.image.draw_bounding_boxes(image_data_with_box,bbox)
        plt.figure(1)
        plt.imshow(image_data_with_box.eval().reshape(origin_shape))
        # 处理后的图像显示窗口
        plt.figure(2)
        # 运行六次获得不同的处理结果
        for i in range(6):
            result = preprocess_for_train(image_data,300,300,bbox)
            plt.subplot(2,3,i+1)
            plt.imshow(result.eval())
        plt.show()

