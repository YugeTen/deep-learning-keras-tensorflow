import numpy as np
import datetime

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from numpy import nan

now = datetime.datetime.now
batch_size = 128
n_classes = 5
n_epoch = 5

img_rows, img_cols = 28, 28
n_filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, n_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = np_utils.to_categorical(train[1], n_classes)
    y_test = np_utils.to_categorical(test[1], n_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    model.fit(x_train, y_train,
              batch_size = batch_size,
              nb_epoch = n_epoch,
              verbose = 1,
              validation_data = (x_test, y_test)
              )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Training time: %s' % (now()-t))
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train_lt5 = x_train[y_train<5]
y_train_lt5 = y_train[y_train<5]
x_test_lt5 = x_test[y_test<5]
y_test_lt5 = y_test[y_test<5]

x_train_gt5 = x_train[y_train>=5]
y_train_gt5 = y_train[y_train>=5] - 5 # make classes start at 0 for np_utils.to_categorical
x_test_gt5 = x_test[y_test>=5]
y_test_gt5 = y_test[y_test>=5] - 5

feature_layers = [
    Convolution2D(n_filters,
                  kernel_size,
                  kernel_size,
                  border_mode = 'valid',
                  input_shape = input_shape),
    Activation('relu'),
    Convolution2D(n_filters, kernel_size, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(0.25),
    Flatten()]
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(n_classes),
    Activation('softmax')
]
model = Sequential(feature_layers+classification_layers)

# train_model(model, (x_train_lt5,y_train_lt5), (x_test_lt5, y_test_lt5), n_classes)
####################### below is how you do fine tuning:

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False
# transfer: train dense layers for new classification task [5...9]
train_model(model, (x_train_gt5,y_train_gt5), (x_test_gt5, y_test_gt5), n_classes)
