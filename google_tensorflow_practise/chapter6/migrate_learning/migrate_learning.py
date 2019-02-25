# -*- coding:utf-8
"""
迁移学习，将一个问题上训练好的模型通过简单的调整使得它使用与一个新的问题

这里使用谷歌训练好的inception-v3模型来区分不同花的类别
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3 模型中代表瓶颈层果的张量名称。在谷歌提供的Inception-v3模型中，这个
# 张量的名称就是'pool_3/_reshape:0
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的Inception-v3模型文件目录。
MODEL_DIR = './inception_v3/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算
# 得到的特征向量保存在文件中，免去重复计算，下面定义这些文件的存放地址
CACHE_DIR = '/tmp/bottleneck'

# 图片数据文件夹。在这个文件夹中每一个文件夹代表一个需要区别的类别，每个子文件夹中存放了对应类别的图片
INPUT_DATA = './flower_photos'

# 验证的数据的百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据的百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


def create_image_lists(testing_precentage, validation_precentage):
    """数据文件夹中读取所有的图片列表并按照训练、验证、测试数据分开
    :param testing_precentage: 测试数据百分比
    :param validation_precentage: 验证数据百分比
    :return: 返回result字典。result存放着不同类别下的验证、测试、训练数据图片名
    """
    result = {}
    # 获取当前目录下的所有子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 　除去第一个目录,因为第一个目录是当前目录
    sub_dirs.pop(0)
    for sub_dir in sub_dirs:
        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名称获取类别的名称
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据及和验证数据集合
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练数据集合、测试、验证
            chance = np.random.randint(100)
            if chance < validation_precentage:
                validation_images.append(base_name)
            elif chance < (testing_precentage + validation_precentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果点
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_precentage
        }
    # 返回整理好的所有数据
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    """　通过类别名称、所属数据集和图片编号获取一张图片的地址
    :param image_lists: 所有图片信息
    :param image_dir: 根目录。存放图片数据的根目录和存放图片特征向量的根目录地址不同
    :param label_name: 类别名称
    :param index: 图片编号
    :param category: 需要获取的是训练数据集、测试数据还是验证数据
    :return:
    """
    # 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中额全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名。
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址位数据根目录的地址加上类别的文件夹再加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    """　通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址。
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :return:
    """
    return get_image_path(image_lists, CACHE_DIR,
                          label_name, index, category) + ".txt"


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """ 使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
    这个过程实际上就是将当前图片组委输入计算瓶颈张量的值。这瓶颈张量的值就是这张图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return: 到达瓶颈层的特征向量
    """
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果拉升为一位数组
    bottleneck_values =np.squeeze(bottleneck_values)
    return bottleneck_values

def get_or_create_bottleneck(
        sess,image_lists,label_name,index,
        category,jpeg_data_tensor,bottleneck_tensor):
    """

    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    """


if __name__ == '__main__':
    result = create_image_lists(testing_precentage=TEST_PERCENTAGE,
                                validation_precentage=VALIDATION_PERCENTAGE)
    with open("./result.txt", 'w') as f:
        f.write(str(result))
