"""
keras 学习，顺序模型
数字十分类问题
"""
import keras
from keras.models import  Sequential
from keras.layers import Dense
import numpy as np

# 建立数据集合
x_data = np.random.random((1000,100))
labels = np.random.randint(10,size=(1000,1))
# 将labels转为one-hot编码
y_data = keras.utils.to_categorical(labels,num_classes=10)


# 建立神经网络模型
model = Sequential()
model.add(Dense(256,activation='relu',input_dim=100))
model.add(Dense(10,activation='softmax'))
# 配置学习参数
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_data,y_data,batch_size=100,epochs=1000)



