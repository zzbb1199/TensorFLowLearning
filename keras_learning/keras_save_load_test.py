"""
keras保存模型和加载模型的方法
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# 划分Ｘ和Ｙ
X = dataset[:,0:8]
Y = dataset[:,8]
# 创建模型
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 创建保存节点
filepath = 'weights-improvements-{epoch:02d}-{val_acc:.2f}.hd5'
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,
                             save_best_only=True,mode='max')
callback_list = [checkpoint]
# Fit the model
model.fit(X,Y,validation_split=0.33,epochs=150,batch_size=10,callbacks=callback_list,
          verbose=0)

