"""
mnist 样本测试
由于mnist样本十分出名，Tensorflow已经将它封装，可以直接使用Tensorflow提供的api
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNISt_data/", one_hot=True)

# 打印training data size:
print('Training data size', mnist.train.num_examples)

# validation data size:
print('Validation data size', mnist.validation.num_examples)

# Testing data size:
print('Testing data size', mnist.test.num_examples)

# 打印example traing data
print('example traning data', mnist.train.images[0])

# 打印example training data label
print('example training data label', mnist.train.labels[0])

# mnist train next batch
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print('x shape', xs.shape) # 784 = 28*28 即一张图片是由28*28个像素组成
print('y shape', ys.shape)
