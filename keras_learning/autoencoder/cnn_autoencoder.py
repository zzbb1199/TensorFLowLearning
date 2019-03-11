"""
卷积自编码器
"""
from keras.datasets import mnist
from keras.models import Model, Sequential, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import numpy as np
import matplotlib.pyplot as plt

# 配置神经网络
batch_size = 256
epochs = 50

# load data
(x_train, _), (x_test, _) = mnist.load_data()

# change value type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# scale the training and testing data to range between 0 and 1
max_value = float(x_train.max())
x_train /= max_value
x_test /= max_value

# reshape 操作
x_train = np.reshape(x_train, (60 * 1000, 28, 28, 1))
x_test = np.reshape(x_test, (10 * 1000, 28, 28, 1))

# 卷积自编码器模型
autoencoder = Sequential()

# 加入卷积编码层
autoencoder.add(Conv2D(16, (3, 3),
                       activation='relu', padding='same',
                       input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(2, 2, padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same'))

# 为了可视化，这里先加入了Flatten
# 为了后续可以继续训练，还原到上一层的shape
autoencoder.add(Flatten())
autoencoder.add(Reshape((4, 4, 8)))

# 加入卷积解码层
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# 获取编码层输出
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer(name='flatten_1').output)

# 配置
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['accuracy'])
# 训练
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                validation_data=(x_test, x_test))
# 输出
num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(16, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


