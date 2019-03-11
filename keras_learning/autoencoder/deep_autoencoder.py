"""
deep autoencoder sample code
"""
import keras
from keras.datasets import mnist
from keras.models import Model,Sequential,Input
from keras.layers import Dense
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

# the origin mnist data is 3D, in order to use it into nn, we need to reshap it to 2D
x_train = np.reshape(x_train, (len(x_train), np.prod(x_train.shape[1:])))
x_test = np.reshape(x_test, (len(x_test), np.prod(x_test.shape[1:])))

# input_dim
input_dim = x_train.shape[1]    # 784
# compress 28*23 to 32
encoding_dim = 32
# calc compression factor
compressing_factor = float(input_dim / encoding_dim)
print('Compression factor', compressing_factor)

# 创建模型
autoencoder = Sequential()

# 添加编码层
autoencoder.add(Dense(128,activation='relu',input_dim=input_dim))
autoencoder.add(Dense(64,activation='relu'))
autoencoder.add(Dense(32,activation='relu'))

# 添加解码层
autoencoder.add(Dense(64,activation='relu'))
autoencoder.add(Dense(128,activation='relu'))
autoencoder.add(Dense(input_dim,activation='sigmoid'))

autoencoder.summary()

# 得到encoder model，方便后续测试
input_imgs = Input(shape=(input_dim,))
encode_layer1 = autoencoder.layers[0]
encode_layer2 = autoencoder.layers[1]
encode_layer3 = autoencoder.layers[2]
encoder = Model(inputs=input_imgs,outputs=encode_layer3(encode_layer2(encode_layer1(input_imgs))))
encoder.summary()


# 配置训练信息
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# 开始训练
autoencoder.fit(x_train,x_train,batch_size=batch_size,epochs=epochs,
                validation_data=(x_test,x_test))


# 查看训练效果
num_image = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0],size=num_image)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18,4))

for i,image_idx in enumerate(random_test_images):
    # 绘制原图
    ax = plt.subplot(3,num_image,i+1)
    plt.imshow(x_test[image_idx].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 绘制编码后的图形
    ax = plt.subplot(3,num_image,i+1+num_image)
    plt.imshow(encoded_imgs[image_idx].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 绘制解码后的图形
    ax = plt.subplot(3, num_image, i + 1 + 2*num_image)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
