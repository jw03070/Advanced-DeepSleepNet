#######################Change 2D_CNN MNIST to 3000 by 6 #####################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import Data_Loader

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Concatenate, Reshape,GlobalAveragePooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Bidirectional


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
x_train = np.array(x_train)
y_train = np.array(y_train)
# x_test, y_test = Data_Loader.Test_Data(1)
x_test = np.array(x_test)
y_test = np.array(y_test)

#print(y_test)

#input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 5
epochs = 20

model = Sequential()
model.add(Conv2D(input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]), kernel_size=(5, 5), strides=(1, 1), padding='same',filters = 50,
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same')) #256
#model.add(Conv2D(256, (5, 5), activation='relu', padding='same')) #256
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(512, (2, 2), activation='relu', padding='same')) #512
#model.add(Reshape((750, 512))) #TEMP
#model.add(Conv2D(512, (2, 2), activation='relu', padding='same')) #512
#model.add(MaxPooling2D(pool_size=(2, 2)))#TEMP
model.add(Dropout(0.25))
#model.add(Reshape((750, 512)))

####################TEMP AREA###################################################
'''
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
#model3.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(GlobalAveragePooling1D())
'''
####################TEMP AREA###################################################

model.add(Flatten())
#################################LSTM 20200810#################################


#################################LSTM 20200810#################################

#model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))

model.add(Dense(500, activation='relu'))
#model.add(Dense(256, activation='relu'))


model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
adam = optimizers.Adam(lr = 0.0001)

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
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_split=0.2)

y_pred = model.predict(x_test)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.savefig('./'+str(fignum)+'Acc.png', dpi=300)
#plt.show()


matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(y_test.argmax(axis=1))
print(y_pred.argmax(axis=1))

#cm = confusion_matrix(y_test, y_pred)


loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
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