import numpy as np
from numpy import nan
import datetime
import matplotlib.pyplot as plt
import os

from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf


def train_model(model, train, test, n_classes,batch_size,n_epoch):
    x_train = train.astype('float32')
    x_test = test.astype('float32')
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

    model.fit(x_train, y_train,
              batch_size = batch_size,
              nb_epoch = n_epoch,
              verbose = 1,
              validation_data = (x_test, y_test)
              )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def reshape_for_retrain(data, deep_img_shape):
    deep_img_rows = deep_img_shape[0]
    deep_img_cols = deep_img_shape[1]
    dataset_img_rows = data.shape[1]
    dataset_img_cols = data.shape[2]
    data = np.pad(data,
                 ((0, 0),
                  ((deep_img_rows - dataset_img_rows) // 2, (deep_img_rows - dataset_img_rows) // 2),
                  ((deep_img_cols - dataset_img_cols) // 2, (deep_img_cols - dataset_img_cols) // 2),
                  (0, 0)),
                 mode='constant',
                 constant_values=1)
    return data
