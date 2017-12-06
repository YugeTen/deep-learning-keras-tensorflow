import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train: 60000x28x28      x_test: 10000x28x28
# y_train: 60000            y_test: 10000
X_test_orig = x_test

# data preparation: preprocessing image data x
from keras import backend as K
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    shape_ord = (1, img_rows, img_cols)
else:
    shape_ord = (img_rows, img_cols, 1)

x_train = x_train.reshape((x_train.shape[0],) + shape_ord) # 60000x1x28x28 (channel_fist)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape((x_test.shape[0],) + shape_ord)
x_test = x_test.astype('float32')
x_test /=255

# data preparation for set task of recognising 6
# test
y_test = y_test == 6
y_test = y_test.astype(int)
# train
x_six = x_train[y_train==6].copy()
y_six = y_train[y_train==6].copy()
y_six = y_six.astype(int)

x_not_six = x_train[y_train!=6].copy()
np.random.seed(0)
np.random.shuffle(x_not_six)
x_not_six = x_not_six[0:6000]
y_not_six = y_train[y_train!=6].copy()
np.random.seed(0)
np.random.shuffle(y_not_six)
y_not_six = y_not_six[0:6000]
y_not_six = y_not_six.astype(int)

x_train = np.append(x_six,x_not_six,axis=0)
y_train = np.append(y_six,y_not_six)
print(x_train.shape)
print(x_test.shape)

n_classes = 2
y_test = y_test == 6
y_train = y_train == 6
y_test = np_utils.to_categorical(y_test, n_classes)
y_train = np_utils.to_categorical(y_train, n_classes)

## build CNN
n_epoch = 20
batch_size = 64
n_filters = 32 # number of kernels
n_pool = 2
n_conv = 3 # kernel size
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# model definition
model = Sequential()
# have to define input_shape for only the first layer
model.add(Conv2D(n_filters, (n_conv, n_conv), padding='valid',input_shape=shape_ord))
model.add(Activation('relu'))
model.add(Conv2D(n_filters, (n_conv, n_conv)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

# compile
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# fit
hist = model.fit(x_train, y_train, batch_size=batch_size,
                 epochs = n_epoch, verbose=1,
                 validation_data=(x_test, y_test))

# evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)



print('Test score:', loss)
print('Test accuracy:', accuracy)