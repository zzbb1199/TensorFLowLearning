"""
使用keras实现MLP on mnist
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# 创建训练数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 对数据进行预处理
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))
# 转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train,
                                     num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test,
                                    num_classes=num_classes)

# 构建MLP网络
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# 打印构建的网络信息
model.summary()
# 配置
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(x_test, y_test))

# 将训练好的模型进行最后的测试
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('the final loss is %f, and the final accuracy is %f' % (score[0],score[1]))
