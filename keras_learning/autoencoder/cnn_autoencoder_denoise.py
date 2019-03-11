"""
CNN 自编码器去噪
"""
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt

# 读取原始数据
(x_train, _), (x_test, _) = mnist.load_data()
# 更改数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# reshape
x_train = np.reshape(x_train, (60 * 1000, 28, 28, 1))
x_test = np.reshape(x_test, (10 * 1000, 28, 28, 1))
# 数据范围压缩到0-1
x_train /= 255
x_test /= 255
# 加入噪声
noise_train = np.random.normal(loc=0.0, scale=0.5, size=x_train.shape)
noise_test = np.random.normal(loc=0.0, scale=0.5, size=x_test.shape)
x_train_noise = x_train + noise_train
x_test_noise = x_test + noise_test
# 裁剪
x_train_noise = x_train_noise.clip(0., 1.)
x_test_noise = x_test_noise.clip(0., 1.)

# 构建自编码器
autoencoder = Sequential()
# Encoder Layers
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder Layers
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# 打印网络结构
autoencoder.summary()

# 配置
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
# 训练
autoencoder.fit(x_train_noise, x_train, batch_size=256, epochs=40,
                validation_data=(x_test_noise, x_test))

# 展示去燥效果
# sample 10个样本出来测试
num_test = 10
random_idx = np.random.randint(x_test.shape[0], size=num_test)

image_prediction = autoencoder.predict(x_test_noise[random_idx])

# 绘图初始画
plt.figure(1)
for i, image_idx in enumerate(random_idx):
    ax = plt.subplot(2, num_test, i + 1)
    plt.imshow(x_test_noise[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, num_test, i + 1 + num_test)
    plt.imshow(image_prediction[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
