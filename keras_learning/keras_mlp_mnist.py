"""
Keras mlp on mnist
"""
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# convert data type to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# reshape the origin data
x_train = np.reshape(x_train,(60*1000,28*28))
x_test = np.reshape(x_test,(10*1000,28*28))
# change the range of data between 0 and 1
x_train /= 255
x_test /= 255
# generate one-hot labels
y_train = keras.utils.to_categorical(y_train,num_classes=10)
y_test = keras.utils.to_categorical(y_test,num_classes=10)

# 建立网络结构
model = Sequential()
model.add(Dense(256,activation='relu',input_dtype=(28*28,)))
model.add(Dense(10,activation='softmax'))

# 配置编译信息
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# 保存模型
# plot_model(model,to_file='model.png')
# 开始训练
history = model.fit(x_train,y_train,batch_size=256,epochs=30,
                    validation_data=(x_test,y_test))
# 测试最终结果
score = model.evaluate(x_test,y_test,batch_size=256)

# save model
# method 1: save model arch and weight(contains biases) respectively
model.save_weights('model_weights.h5')
with open('model_arch.json','w') as f:
    f.write(model.to_json())

# method 2: save entire model
model.save('model.h5')

# 绘制训练过程
# 精度
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# show
plt.show()
