#######################Change 2D_CNN MNIST to 3000 by 6 #####################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import Data_Loader

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, concatenate, Reshape,GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model


import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras import layers
from keras.optimizers import RMSprop
from keras import backend

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

img_rows = 3000
img_cols = 6

#(x_train, y_train), (x_test, y_test) = #"NEED to change"# #keras.datasets.mnist.load_data()

#if 0 there is no test dataset
Test_number = 1
fignum = Test_number
x_train,y_train,x_test,y_test = Data_Loader.Data(1,Test_number)
x_train = np.array(x_train)  #(41316,6,3000)
y_train = np.array(y_train)
# x_test, y_test = Data_Loader.Test_Data(1)
x_test = np.array(x_test)    #(839,6,3000)
y_test = np.array(y_test)

#############################
raw_x_train = x_train[:,0:1,:] #(41316,1,3000)
wake_x_train = x_train[:,1:2,:]
rem_x_train = x_train[:,2:3,:]
n1_x_train = x_train[:,3:4,:]
n2_x_train = x_train[:,4:5,:]
n3_x_train = x_train[:,5:6,:]

raw_x_test = x_test[:,0:1,:]
wake_x_test = x_test[:,1:2,:]
rem_x_test = x_test[:,2:3,:]
n1_x_test = x_test[:,3:4,:]
n2_x_test = x_test[:,4:5,:]
n3_x_test = x_test[:,5:6,:]

#################################
raw_x_train = raw_x_train.reshape(raw_x_train.shape[0], img_rows, 1)
wake_x_train = wake_x_train.reshape(wake_x_train.shape[0], img_rows, 1)
rem_x_train = rem_x_train.reshape(rem_x_train.shape[0], img_rows, 1)
n1_x_train = n1_x_train.reshape(n1_x_train.shape[0], img_rows, 1)
n2_x_train = n2_x_train.reshape(n2_x_train.shape[0], img_rows, 1)
n3_x_train = n3_x_train.reshape(n3_x_train.shape[0], img_rows, 1)

raw_x_test = raw_x_test.reshape(raw_x_test.shape[0], img_rows, 1)
wake_x_test = wake_x_test.reshape(wake_x_test.shape[0], img_rows, 1)
rem_x_test = rem_x_test.reshape(rem_x_test.shape[0], img_rows, 1)
n1_x_test = n1_x_test.reshape(n1_x_test.shape[0], img_rows, 1)
n2_x_test = n2_x_test.reshape(n2_x_test.shape[0], img_rows, 1)
n3_x_test = n3_x_test.reshape(n3_x_test.shape[0], img_rows, 1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 5
epochs = 20

model1 = Sequential()
model1.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model1.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model1.add(MaxPooling1D(pool_size=2, strides=2))
model1.add(Dropout(0.2))
model1.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model1.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model1.add(MaxPooling1D(pool_size=2, strides=2))
model1.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model1.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model1.add(MaxPooling1D(pool_size=2, strides=2))
model1.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model1.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model1.add(MaxPooling1D(pool_size=2, strides=2))
model1.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model1.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model1.add(MaxPooling1D(pool_size=2, strides=2))
model1.add(Flatten())

model2 = Sequential()
model2.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model2.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model2.add(MaxPooling1D(pool_size=2, strides=2))
model2.add(Dropout(0.2))
model2.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model2.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model2.add(MaxPooling1D(pool_size=2, strides=2))
model2.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model2.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model2.add(MaxPooling1D(pool_size=2, strides=2))
model2.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model2.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model2.add(MaxPooling1D(pool_size=2, strides=2))
model2.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model2.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model2.add(MaxPooling1D(pool_size=2, strides=2))
model2.add(Flatten())

model3 = Sequential()
model3.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model3.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model3.add(MaxPooling1D(pool_size=2, strides=2))
model3.add(Dropout(0.2))
model3.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model3.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model3.add(MaxPooling1D(pool_size=2, strides=2))
model3.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model3.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model3.add(MaxPooling1D(pool_size=2, strides=2))
model3.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model3.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model3.add(MaxPooling1D(pool_size=2, strides=2))
model3.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model3.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model3.add(MaxPooling1D(pool_size=2, strides=2))
model3.add(Flatten())

model4 = Sequential()
model4.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model4.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model4.add(MaxPooling1D(pool_size=2, strides=2))
model4.add(Dropout(0.2))
model4.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model4.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model4.add(MaxPooling1D(pool_size=2, strides=2))
model4.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model4.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model4.add(MaxPooling1D(pool_size=2, strides=2))
model4.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model4.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model4.add(MaxPooling1D(pool_size=2, strides=2))
model4.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model4.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model4.add(MaxPooling1D(pool_size=2, strides=2))
model4.add(Flatten())

model5 = Sequential()
model5.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model5.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model5.add(MaxPooling1D(pool_size=2, strides=2))
model5.add(Dropout(0.2))
model5.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model5.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model5.add(MaxPooling1D(pool_size=2, strides=2))
model5.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model5.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model5.add(MaxPooling1D(pool_size=2, strides=2))
model5.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model5.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model5.add(MaxPooling1D(pool_size=2, strides=2))
model5.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model5.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model5.add(MaxPooling1D(pool_size=2, strides=2))
model5.add(Flatten())

model6 = Sequential()
model6.add(Conv1D(input_shape = (3000,1), kernel_size=5,strides=3,padding='valid',filters = 64,activation='relu'))
model6.add(Conv1D(filters=128,kernel_size = 5,activation='relu'))
model6.add(MaxPooling1D(pool_size=2, strides=2))
model6.add(Dropout(0.2))
model6.add(Conv1D(filters=128,kernel_size = 13,activation='relu'))
model6.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model6.add(MaxPooling1D(pool_size=2, strides=2))
model6.add(Conv1D(filters=256,kernel_size = 7,activation='relu'))
model6.add(Conv1D(filters=64,kernel_size = 4,activation='relu'))
model6.add(MaxPooling1D(pool_size=2, strides=2))
model6.add(Conv1D(filters=32,kernel_size = 3,activation='relu'))
model6.add(Conv1D(filters=64,kernel_size = 6,activation='relu'))
model6.add(MaxPooling1D(pool_size=2, strides=2))
model6.add(Conv1D(filters=8,kernel_size = 5,activation='relu'))
model6.add(Conv1D(filters=8,kernel_size = 2,activation='relu'))
model6.add(MaxPooling1D(pool_size=2, strides=2))
model6.add(Flatten())

merged = concatenate([ model1.output, model2.output, model3.output, model4.output, model5.output, model6.output])
merged = Dense(64)(merged)
merged = Dense(num_classes, activation='softmax')(merged)
model = Model(inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input], outputs=merged)
adam = optimizers.Adam(lr = 1e-4)
model.summary()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1_m,precision_m, recall_m])
'''
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(x_test, y_test))
'''
hist = model.fit([raw_x_train,wake_x_train,rem_x_train,n1_x_train,n2_x_train,n3_x_train], y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_split=0.2)

y_pred = model.predict([raw_x_test,wake_x_test,rem_x_test,n1_x_test,n2_x_test,n3_x_test])

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.savefig('./'+str(fignum)+'Acc.png', dpi=300)
plt.show()


matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(y_test.argmax(axis=1))
print(y_pred.argmax(axis=1))

#cm = confusion_matrix(y_test, y_pred)


loss, accuracy, f1_score, precision, recall = model.evaluate([raw_x_test,wake_x_test,rem_x_test,n1_x_test,n2_x_test,n3_x_test], y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
print('Test f1_score:', f1_score)
print('Test precision:', precision)
print('Test recall:', recall)
print(matrix)
sn.heatmap(matrix, annot=True, annot_kws={"size": 10}, fmt='d')
plt.savefig('./'+str(fignum)+'Confusion.png', dpi=300)

'''
y_test = list(set(y_test))
print(y_test)
y_pred = list(set(y_pred))
print(y_pred)
'''
