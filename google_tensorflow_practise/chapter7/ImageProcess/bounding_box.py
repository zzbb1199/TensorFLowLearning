"""
TensorFlow标注框处理,目前效果很奇怪：裁剪出来的图片和标注的边框不同。
"""
import tensorflow as tf
import matplotlib.pyplot as plt

def func1(image_data):
    """ 处理标注框示例程序
    :return:
    """
    with tf.Session() as sess:
        # 将图像缩小一点,这样可视化能让标注显示得更清楚
        image_data = tf.image.resize_images(image_data,[170,240])
        # tf.image.draw_bounding_boxes输入的是一个batched 四维数组，所以需要将解码之后的图像增添一维
        batched = tf.expand_dims(image_data,0)

        # 给出每一张图像的所有标注框。
        boxes = tf.constant([
            [
                [0.05,0.05,0.9,0.7]
            ]
        ])
        # 加入标注框
        result = tf.image.draw_bounding_boxes(batched,boxes)
        # 要想显示加入标注后的图像，需要reshape为3维

def func2(image_data):
    """随机截取图像上有信息含量的部分
    """
    with tf.Session() as sess:
        # 保留原始图片的shape，方便后续显示图片时调用
        origin_shape = sess.run(image_data).shape
        boxes = tf.constant([[
            [0,0,0.9,0.9]
        ]])
        # 可以通过提供标注框的方式来告诉随机截取图像的算法那些部分是“有信息量”的
        begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(image_data),boxes
        )
        # 通过标注框可视化随机截取得到的图像x
        batched = tf.expand_dims(
            image_data,0
        )
        plt.figure()
        image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
        plt.subplot(1,2,1)
        plt.imshow(image_with_box.eval().reshape(origin_shape))
        # 截取随机出来的图像。
        distorted_image = tf.slice(image_data,begin,size)
        plt.subplot(1,2,2)
        # 显示处理之后的图片
        plt.imshow(distorted_image.eval())
        plt.show()
if __name__ == '__main__':
    # 读取原始图片
    image_data_raw = tf.gfile.FastGFile("./picture/pic.jpg", 'rb').read()
    # 解码
    image_data = tf.image.decode_jpeg(image_data_raw)
    # 转换数据格式
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    func2(image_data)