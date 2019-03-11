"""
keras 函数响应式编程
实现全连接网络
"""
import keras
from keras.models import Model
from keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

# 训练数据
x_data = np.linspace(-1, 1, 1000)
y_data = 2 * x_data ** 2 + 3 * x_data + 4

# 定义网络结构
inputs = Input(shape=(1,))
x = Dense(32, activation='linear')(inputs)
outputs = Dense(1, activation='linear')(x)

# 生成模型
model = Model(inputs=inputs, outputs=outputs)
# 配置训练
model.compile(
    optimizer='rmsprop',
    loss=keras.losses.mse,
    metrics=['accuracy']
)
# 开始训练
history = model.fit(x_data, y_data,epochs=100)
