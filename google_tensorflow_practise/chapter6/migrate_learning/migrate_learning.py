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

# Inception-v3 模型中代表瓶颈层结果的张量名称。


