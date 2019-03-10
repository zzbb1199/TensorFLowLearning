"""
Keras将保存好的模型导入
"""
import keras
from keras.datasets import  mnist
from keras.models import model_from_json,load_model
import numpy as np
# (_,_),(x_test,y_test) = mnist.load_data()
# x_test = x_test.astype('float32')
# x_test = np.reshape(x_test,(10*1000,28*28))
# x_test /= 255
#
# # method 1 : load model and weights respectively
# with open('model_arch.json','r') as f:
#     model = model_from_json(f.read())
# model.load_weights('model_weights.h5')
#
# # 预测
# y_prediction = model.predict(x_test)
# y_prediction = np.argmax(y_prediction,axis=1)
#
# # 比较精度
# accur = np.mean(
#     np.equal(y_prediction,y_test)
# )
# print(accur)
#
# # 测试第二种方法
# del model
# model = load_model('model.h5')
# y_prediction = model.predict(x_test)
# y_prediction = np.argmax(y_prediction,axis=1)
# accu = np.mean(np.equal(y_prediction,y_test))
# print(accur)
model = load_model('weights-improvements-135-0.78.hd5')
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# 划分Ｘ和Ｙ
X = dataset[:,0:8]
Y = dataset[:,8]
print(Y)
Y_prediction = model.predict(X)
Y_prediction = Y_prediction.reshape(Y.shape)
Y_prediction[Y_prediction > 0.5] = 1
Y_prediction[Y_prediction <= 0.5] = 0

# 获取精度
print(np.mean(
    np.equal(Y_prediction,Y)
))