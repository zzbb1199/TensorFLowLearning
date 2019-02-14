"""
mnist CNN实现
"""
import tensorflow as tf

IMAGE_SIZE = 32
NUM_CHANNEL = 3     # 3通道
BATCH_SIZE = 100

x = tf.placeholder(tf.float32,
                   shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL),name="x-input")
